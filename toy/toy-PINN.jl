using Flux, Zygote, ForwardDiff, Random, Optim
using DifferentialEquations, OrdinaryDiffEq
using Plots
using Flux.Optimise: update!
using BSON: @save, @load
using ProgressBars
using Printf
using LinearAlgebra
using Statistics
using DiffEqFlux

n_epoch = 1000
nplot = 100
npoint = 100
ns = 5
opt = ADAMW(0.005, (0.9, 0.999), 1.f-6)

function f(y, k, t)
    dydt = [-2 * k[1] * y[1]^2 - k[2] * y[1], 
            k[1] * y[1]^2 - k[4] * y[2] * y[4],
            k[2] * y[1] - k[3] * y[3],
            k[3] * y[3] - k[4] * y[2] * y[4],
            k[4] * y[2] * y[4]]
    return dydt
end

u0 = Float32[1, 1, 0, 0, 0]
k = Float32[0.1, 0.2, 0.13, 0.3]
tspan = Float32[0, 40.0]
dt = tspan[2] / npoint
ts = tspan[1]:dt:tspan[2]

prob = ODEProblem(f, u0, tspan, k)
sol = solve(prob, saveat=ts);

plt = scatter(sol, yscale=:identity);
png(plt, "figs/sol");

hls = 8;
phi_ = Flux.Chain(Dense(1, hls, tanh),
                Dense(hls, hls, tanh),
                Dense(hls, hls, tanh),
                Dense(hls, hls, tanh),
                Dense(hls, ns))

θ, re = Flux.destructure(phi_)

function phi(t, θ)
    return u0 .+ t .* re(θ)([t])
end
phi(1, θ)

dfdx = (t, θ) -> ForwardDiff.derivative(t -> phi(t, θ), t)
dfdx(1, θ)

function inner_loss(t, θ)
    mean(abs2, dfdx(t, θ) - f(phi(t, θ), k, t))
end

loss(θ) = mean(abs2, [inner_loss(t, θ) for t in ts])

l_loss_train = []
l_loss_val = []
l_grad = []
iter = 1
cb = function (p, loss_train, loss_val, g_norm)
    global iter
    global l_loss_train, l_loss_val, l_grad, iter
    push!(l_loss_train, loss_train)
    push!(l_loss_val, loss_val)
    push!(l_grad, g_norm)

    if iter % nplot == 0
        pred = hcat([phi(t, θ) for t in ts]...)
        l_plt = []
        for i in 1:ns
            plt = scatter(ts, sol[i, :], label="label");
            plot!(plt, ts, pred[i, :], label="PINN")
            xlabel!(plt, "Time")
            ylabel!(plt, "x$i")
            push!(l_plt, plt)
        end
        plt_all = plot(l_plt..., legend=:best, size=(800, 800))
        png(plt_all, "figs/pinn")

        plt_loss = plot(l_loss_train, xscale=:identity, yscale=:log10, label="Training")
        plot!(plt_loss, l_loss_val, xscale=:identity, yscale=:log10, label="Validation")
        plt_grad = plot(l_grad, xscale=:identity, yscale=:log10, label="grad_norm")
        xlabel!(plt_loss, "Epoch")
        xlabel!(plt_grad, "Epoch")
        ylabel!(plt_loss, "Loss")
        ylabel!(plt_grad, "Gradient Norm")
        plt_all = plot([plt_loss, plt_grad]..., legend=:top)
        png(plt_all, "figs/loss_grad")
    end
    iter = iter + 1
end

epochs = ProgressBar(1:n_epoch);
for epoch in epochs
    global θ
    l = loss(θ)
    grad = ForwardDiff.gradient(x -> loss(x), θ)
    update!(opt, θ, grad)
    set_description(epochs, string(@sprintf("Loss %.4e", l)))
    cb(θ, l, l, norm(grad, 2))
end