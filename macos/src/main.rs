use std::env;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::thread;
use std::time::{Duration, Instant};

use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoop};
use tao::window::WindowBuilder;
use wry::WebViewBuilder;

const PORT: u16 = 8765;

fn resource_dir() -> PathBuf {
    let exe = env::current_exe().expect("cannot locate executable");
    // Inside .app bundle: Contents/MacOS/subtitle-anywhere
    // Resources at:       Contents/Resources/
    exe.parent()
        .expect("no parent")
        .parent()
        .expect("no Contents")
        .join("Resources")
}

fn start_server(res: &Path) -> Child {
    let python = res.join("python/bin/python3");
    let web_py = res.join("app/server/web.py");

    Command::new(&python)
        .arg(&web_py)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(PORT.to_string())
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .spawn()
        .unwrap_or_else(|e| {
            panic!("failed to start python server: {e}\npython: {python:?}\nweb.py: {web_py:?}")
        })
}

fn wait_for_server(timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if TcpStream::connect(("127.0.0.1", PORT)).is_ok() {
            return true;
        }
        thread::sleep(Duration::from_millis(100));
    }
    false
}

#[allow(clippy::zombie_processes)] // event_loop.run never returns; server is killed in CloseRequested
fn main() {
    let res = resource_dir();
    let mut server = start_server(&res);

    if !wait_for_server(Duration::from_secs(30)) {
        eprintln!("timeout waiting for python server on port {PORT}");
        let _ = server.kill();
        let _ = server.wait();
        std::process::exit(1);
    }

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Subtitle Anywhere")
        .with_inner_size(tao::dpi::LogicalSize::new(900.0, 700.0))
        .build(&event_loop)
        .expect("failed to build window");

    let _webview = WebViewBuilder::new()
        .with_url(format!("http://127.0.0.1:{PORT}"))
        .build(&window)
        .expect("failed to build webview");

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        if let Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } = event
        {
            let _ = server.kill();
            let _ = server.wait();
            *control_flow = ControlFlow::Exit;
        }
    });
}
