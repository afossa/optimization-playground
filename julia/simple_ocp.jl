#=
Example 5.1 in F. Topputo and C. Zhang,
“Survey of Direct Transcription for Low-Thrust Space Trajectory Optimization with Applications,”
Abstract and Applied Analysis, vol. 2014, pp. 1–15, 2014, doi: 10.1155/2014/851720.
=#

using AppleAccelerate

using OrdinaryDiffEq
using StaticArrays

using InfiniteOpt
using Ipopt

using OptimalControl
using NLPModelsIpopt

using GLMakie

# time horizon
t0 = 0.0;
tf = 1.0;
t_span = [t0, tf];
nb_nodes = 101;

# initial conditions
x0 = SA[1.0, 0.0];

# analytical solution
analytic_control(t) = (-(tanh.(1.0 .- t) .+ 0.5) .* cosh.(1.0 .- t)) ./ cosh(1.0);

t_analytic = range(t0, tf, nb_nodes);
u_analytic = analytic_control(t_analytic);
x1_analytic = cosh.(1.0 .- t_analytic) ./ cosh(1.0);

# explicit integration using the analytical optimal control law
function dynamics(x::AbstractVector, u::Function, t::Number)
    u_curr = u(t)
    return SA[
        0.5*x[1]+u_curr,
        u_curr^2+x[1]*u_curr+(5.0/4.0)*x[1]^2
    ]
end;

ode_prob = ODEProblem(dynamics, x0, t_span, analytic_control);
ode_sol = solve(ode_prob, Vern9(), abstol=1e-14, reltol=1e-14);
x_explicit = cat(ode_sol(t_analytic).u..., dims=2);
@assert x_explicit[1, end] ≈ x1_analytic[end]

zero_prob = remake(ode_prob, p=t -> 0.0);
zero_sol = solve(zero_prob, Vern9(), abstol=1e-14, reltol=1e-14);

# optimal control problem with InfiniteOpt
nlp_model = InfiniteModel(Ipopt.Optimizer);
set_optimizer_attribute(nlp_model, "print_user_options", "yes");
set_optimizer_attribute(nlp_model, "tol", 1e-12);

# time variable
@infinite_parameter(
    nlp_model,
    t in t_span,
    num_supports = nb_nodes,
    derivative_method = OrthogonalCollocation(3)
);

# state and control variables
@variable(nlp_model, x[1:2], Infinite(t));
@variable(nlp_model, u, Infinite(t));
constant_over_collocation(u, t);

# dynamical constraints
@constraint(nlp_model, ∂(x[1], t) == 0.5 * x[1] + u);
@constraint(nlp_model, ∂(x[2], t) == u^2 + x[1] * u + (5.0 / 4.0) * x[1]^2);

# initial conditions
@constraint(nlp_model, x[1](t0) == x0[1]);
@constraint(nlp_model, x[2](t0) == x0[2]);

# objective function
@objective(nlp_model, Min, x[2](tf));

# initial guess
set_start_value_function(x[1], t -> zero_sol(t)[1]);
set_start_value_function(x[2], t -> zero_sol(t)[2]);

optimize!(nlp_model);

t_inf_opt = value(t);
x_inf_opt = value(x);
u_inf_opt = value(u);

# optimal control problem with OptimalControl
ocp = @def begin
    # time, state, and control variables
    t ∈ [t0, tf], time
    x ∈ R², state
    u ∈ R, control

    # dynamics
    ẋ(t) == [0.5 * x[1](t) + u(t), u(t)^2 + x[1](t) * u(t) + (5.0 / 4.0) * x[1](t)^2]

    # boundary conditions
    x(t0) == x0

    # objective
    x[2](tf) → min
end;

initial_guess = @init ocp begin
    u(t) := 0.0
    x(collect(t_analytic)) := zero_sol(t_analytic).u
end;

sol_ocp = solve(
    ocp;
    grid_size=nb_nodes,
    scheme=:gauss_legendre_3,
    tol=1e-12,
    init=initial_guess
);

t_ocp = time_grid(sol_ocp);
x_ocp = cat(state(sol_ocp).(t_ocp)..., dims=2);
u_ocp = control(sol_ocp).(t_ocp);

# trajectory plot
fig1 = Figure();
ax11 = Axis(fig1[1, 1], xlabel=L"t", ylabel=L"x_1(t)");
lines!(ax11, t_analytic, x1_analytic, color=:blue, linewidth=2, label="analytic");
scatter!(ax11, t_inf_opt, x_inf_opt[1], color=:red, marker=:x, label="InfiniteOpt");
scatter!(ax11, t_ocp, x_ocp[1, :], color=:green, marker=:circle, label="OptimalControl");
axislegend(ax11, position=:rt);

ax21 = Axis(fig1[2, 1], xlabel=L"t", ylabel=L"x_2(t)");
lines!(ax21, t_analytic, x_explicit[2, :], color=:blue, linewidth=2, label="analytic");
scatter!(ax21, t_inf_opt, x_inf_opt[2], color=:red, marker=:x, label="InfiniteOpt");
scatter!(ax21, t_ocp, x_ocp[2, :], color=:green, marker=:circle, label="OptimalControl");
axislegend(ax21, position=:rb);

# control history
fig2 = Figure();
ax2 = Axis(fig2[1, 1], xlabel=L"t", ylabel=L"u(t)");
lines!(ax2, t_analytic, u_analytic, color=:blue, linewidth=2, label="analytic");
scatter!(ax2, t_inf_opt[2:end], u_inf_opt[2:end], color=:red, marker=:x, label="InfiniteOpt");
scatter!(ax2, t_ocp, u_ocp, color=:green, marker=:circle, label="OptimalControl");
axislegend(ax2, position=:rb);

display(GLMakie.Screen(), fig1);
display(GLMakie.Screen(), fig2);
