using ForwardDiff
using StaticArrays
using LinearAlgebra
using JuMP
using Ipopt
using UnoSolver

function rosenbrock_1d(x::AbstractVector, p::AbstractVector{<:Real})
    return (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
end;

function rosenbrock_gradient_1d(x::AbstractVector{<:Number}, p::AbstractVector{<:Real})
    return SA[
        2.0*(x[1]-p[1])+4.0*p[2]*x[1]*(x[1]^2-x[2]),
        2.0*p[2]*(x[2]-x[1]^2)
    ]
end;

function rosenbrock_hessian_1d(x::AbstractVector{<:Number}, p::AbstractVector{<:Real})
    return SA[
        (12.0*p[2]*x[1]^2-4.0*p[2]*x[2]+2.0) (-4.0*p[2]*x[1]);
        (-4.0*p[2]*x[1]) (2.0*p[2])
    ]
end;

function rosenbrock_nd(x::AbstractVector{<:Number}, p::AbstractVector{<:Number})
    y1 = p[1] * sum(x .^ 2)
    y2 = 0.0
    for k in 1:length(x)-1
        y2 += (p[2] - x[k])^2 + p[3] * (x[k+1] - x[k]^2)^2
    end
    return SA[y1, y2]
end;

pars = SA[1.0, 100.0];
x0 = SA[-3.0, -4.0];

@assert norm(ForwardDiff.gradient(x -> rosenbrock_1d(x, pars), x0) .- rosenbrock_gradient_1d(x0, pars)) ≈ 0.0
@assert norm(ForwardDiff.hessian(x -> rosenbrock_1d(x, pars), x0) .- rosenbrock_hessian_1d(x0, pars)) ≈ 0.0

model = Model(() -> UnoSolver.Optimizer(preset="ipopt"));
set_attribute(model, "primal_tolerance", 1e-12);
set_attribute(model, "dual_tolerance", 1e-12);
set_attribute(model, "max_iterations", 200);
# Using the Hessian approximation increases the number of iterations from 30 to 182.
set_attribute(model, "hessian_model", "exact");

# Ipopt options
# set_attribute(model, "print_user_options", "yes");
# Using the Hessian approximation increases the number of iterations from 30 to 40.
# set_attribute(model, "hessian_approximation", "limited-memory");
# set_attribute(model, "tol", 1e-12);

@variable(model, x[1:2]);
@objective(model, Min, rosenbrock_1d(x, pars));
set_start_value.(x, x0);

println("Initial guess: ", start_value.(x), "\n");

optimize!(model);
@assert is_solved_and_feasible(model);
@assert norm(value.(x) - SA[pars[1], pars[1]^2]) < 1e-12

println("\nOptimal solution: ", value.(x));
