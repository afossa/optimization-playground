use cvxrust::{
    Problem, SolveStatus,
    atoms::{index, slice, sum_squares},
    constraint,
    expr::{constant, variable},
    solver::Settings,
};
use nalgebra::{Const, DVector, Dyn};
use plotters::{
    chart::ChartBuilder,
    prelude::{IntoDrawingArea, SVGBackend},
    series::LineSeries,
    style::{Color, IntoFont, RGBColor, ShapeStyle, WHITE},
};

/// https://jump.dev/Convex.jl/stable/examples/general_examples/control/
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // physical parameters
    let h = 0.1; // time step
    let m = 1.0; // mass
    let d = 0.1; // drag coefficient
    let g = [0.0, -9.8]; // gravity acceleration

    // optimization parameters
    let nn: usize = 100;
    let ns = nn - 1;
    let mu = 0.0; // penalty

    // boundary conditions
    let x0 = [0.0, 0.0, -20.0, 100.0];
    let xf = [100.0, 100.0, 0.0, 0.0];

    // row-major storage
    let x = variable((4, nn));
    let u = variable((2, ns));

    let mut constr = Vec::new();

    // boundary conditions
    for i in 0..4 {
        #[allow(unused)]
        constr.push(constraint!((index(&x, nn * i)) == (x0[i])));
        #[allow(unused)]
        constr.push(constraint!((index(&x, nn * (i + 1) - 1)) == (xf[i])));
    }

    // dynamics
    for i in 0..2 {
        let p = slice(&x, nn * i, nn * (i + 1) - 1);
        let v = slice(&x, nn * (i + 2), nn * (i + 3) - 1);
        assert_eq!(p.shape().size(), ns);
        assert_eq!(v.shape().size(), ns);

        let f = slice(&u, ns * i, ns * (i + 1));
        assert_eq!(f.shape().size(), ns);

        let a = &f / m + constant(g[i]) - d * &v;

        let p1 = slice(&x, nn * i + 1, nn * (i + 1));
        let v1 = slice(&x, nn * (i + 2) + 1, nn * (i + 3));
        assert_eq!(p1.shape().size(), ns);
        assert_eq!(v1.shape().size(), ns);

        #[allow(unused)]
        constr.push(constraint!(p1 == (&p + h * &v)));
        #[allow(unused)]
        constr.push(constraint!(v1 == (&v + h * &a)));
    }

    // objective
    let v = slice(&x, nn * 2, nn * 4);
    assert_eq!(v.shape().size(), 2 * nn);
    let obj = mu * sum_squares(&v) + sum_squares(&u);

    // optimization problem
    let prob = Problem::minimize(obj.clone()).subject_to(constr).build();

    // solution
    let sol = prob.solve_with(Settings {
        verbose: true,
        ..Default::default()
    })?;
    assert_eq!(sol.status, SolveStatus::Optimal);

    // optimal state and control time series (column-major storage)
    let xv = &sol[&x].clone().reshape_generic(Dyn(nn), Const::<4>);
    let uv = &sol[&u].clone().reshape_generic(Dyn(ns), Const::<2>);
    let ov = mu * xv.view((0, 2), (nn, 2)).norm_squared() + uv.norm_squared();
    println!("\nObjective value: {}", ov);

    // plot trajectory
    let root = SVGBackend::new("figures/trajectory.svg", (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_series = xv.column(0);
    let y_series = xv.column(1);
    let x_limits = (x_series.min(), x_series.max());
    let y_limits = (y_series.min(), y_series.max());
    let x_buf = 0.05 * (x_limits.1 - x_limits.0);
    let y_buf = 0.05 * (y_limits.1 - y_limits.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Optimal trajectory", ("sans-serif", 16).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_limits.0 - x_buf)..(x_limits.1 + x_buf),
            (y_limits.0 - y_buf)..(y_limits.1 + y_buf),
        )?;
    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .axis_desc_style(("sans-serif", 14))
        .draw()?;
    chart.draw_series(LineSeries::new(
        x_series.iter().zip(y_series.iter()).map(|(&x, &y)| (x, y)),
        ShapeStyle {
            color: RGBColor(0x20, 0x5e, 0xa6).to_rgba(),
            filled: true,
            stroke_width: 3,
        },
    ))?;
    root.present()?;

    // plot force magnitude time series
    let root = SVGBackend::new("figures/force_timeseries.svg", (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_series = DVector::from_fn(ns, |i, _| (i as f64) * h);
    let y_series = DVector::from_fn(ns, |i, _| uv.row(i).norm_squared());
    let x_limits = (x_series.min(), x_series.max());
    let y_limits = (y_series.min(), y_series.max());
    let x_buf = 0.05 * (x_limits.1 - x_limits.0);
    let y_buf = 0.05 * (y_limits.1 - y_limits.0);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Force magnitude time series",
            ("sans-serif", 16).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_limits.0 - x_buf)..(x_limits.1 + x_buf),
            (y_limits.0 - y_buf)..(y_limits.1 + y_buf),
        )?;
    chart
        .configure_mesh()
        .x_desc("t")
        .y_desc("F^2(t)")
        .axis_desc_style(("sans-serif", 14))
        .draw()?;
    chart.draw_series(LineSeries::new(
        x_series.iter().zip(y_series.iter()).map(|(&x, &y)| (x, y)),
        ShapeStyle {
            color: RGBColor(0xaf, 0x30, 0x29).to_rgba(),
            filled: true,
            stroke_width: 3,
        },
    ))?;
    root.present()?;

    Ok(())
}
