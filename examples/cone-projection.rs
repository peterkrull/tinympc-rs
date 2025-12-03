use std::time::Duration;

use nalgebra::vector;
use rerun::Color;
use tinympc_rs::{CircularCone, Error, project::ProjectSingle};

fn main() -> Result<(), Error> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .without_timestamps()
        .init()
        .ok();

    let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
    let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();
    eprintln!("Run this to view profiling data:  puffin_viewer {server_addr}");
    profiling::puffin::set_scopes_on(true);

    std::thread::sleep(std::time::Duration::from_millis(1000));

    let rec = rerun::RecordingStreamBuilder::new("tinympc-constraints")
        .spawn()
        .unwrap();

    // Velocity limiter
    let mut cone_project = CircularCone::new()
        .axis(vector![1.0, 1.0])
        .vertex(vector![-0.2, -0.2])
        .mu(1.0);

    let mut original = Vec::new();
    let mut projected = Vec::new();
    let mut arrows = Vec::new();

    for k in 0..=100 {
        original.clear();
        projected.clear();
        arrows.clear();

        cone_project = cone_project.mu(k as f32 / 50.0);

        for x in -4..=4 {
            for y in -4..=4 {
                let coord_x = x as f32 / 10.0;
                let coord_y = y as f32 / 10.0;
                original.push((coord_x, coord_y));

                let mut project = vector![coord_x, coord_y];
                cone_project.project(project.as_view_mut());

                projected.push((project.x, project.y));
                arrows.push((project.x - coord_x, project.y - coord_y));
            }
        }

        rec.set_time("time-index", Duration::from_millis(k));

        let component = rerun::Points2D::new(&original)
            .with_colors([Color::from_rgb(255, 180, 0)])
            .with_radii([0.01]);
        rec.log("original", &component).unwrap();

        let component = rerun::Points2D::new(&projected).with_radii([0.01]);
        rec.log("projected", &component).unwrap();

        let component = rerun::Arrows2D::from_vectors(&arrows).with_origins(&original);
        rec.log("arrows", &component).unwrap();
    }

    Ok(())
}
