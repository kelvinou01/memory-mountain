import subprocess
import csv
from pathlib import Path
import typer
import plotly.graph_objects as go
from scipy.interpolate import griddata
import numpy as np
from humanize import naturalsize

app = typer.Typer()

SIZES = [4*1024]
while SIZES[-1] < 64*1024*1024:
    SIZES.append(SIZES[-1]*4)

STRIDES = list(range(1, 33))
RESULT_FILE = Path("bench_results.csv")

def run_bench(size: int, stride: int) -> float | None:
    try:
        r = subprocess.run(
            ["./bench", str(size), str(stride)],
            capture_output=True,
            text=True,
            check=True
        )
        return float(r.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None

@app.command()
def bench():
    """Run benchmark and save results to CSV."""
    with open(RESULT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size_bytes", "stride_elems", "throughput_MBps"])
        for size in SIZES:
            for stride in STRIDES:
                mbps = run_bench(size, stride)
                mbps_val = mbps if mbps is not None else 0.0
                writer.writerow([size, stride, mbps_val])
                print(f"size={size//1024}KB stride={stride} elems -> {mbps_val:.2f} MB/s")
    print(f"Results saved to {RESULT_FILE}")

@app.command()
def visualize(output: str = "plot.html"):
    """Plot 3D throughput from CSV results with Plotly."""
    if not RESULT_FILE.exists():
        typer.echo("Run 'bench' first to generate results.")
        raise typer.Exit(1)
    
    X, Y, Z = [], [], []
    with open(RESULT_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append(int(row["stride_elems"]) * 8)   # stride in bytes
            Y.append(int(row["size_bytes"]))         # size in bytes
            Z.append(float(row["throughput_MBps"])) # MB/s
    
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    
    # Convert Y to log2 scale
    Y_log = np.log2(Y)
    
    # Create a regular grid for surface
    xi = np.linspace(X.min(), X.max(), 50)
    yi = np.linspace(Y_log.min(), Y_log.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((X, Y_log), Z, (Xi, Yi), method='linear')
    
    fig = go.Figure()
    
    # Add surface plot
    fig.add_trace(go.Surface(
        x=Xi[0],
        y=Yi[:, 0],
        z=Zi,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            title="Throughput (MB/s)",
            thickness=15,
            len=0.6,
            x=0.98
        ),
        name='Surface',
        hovertemplate='Stride: %{x:.1f}<br>Size (log2): %{y:.1f}<br>Throughput: %{z:.2f} MB/s<extra></extra>'
    ))
    

    # Add red lines tracing the ridges at cache boundaries
    y_log_192kb = np.log2(192 * 1024)
    y_log_12mb = np.log2(12 * 1024 * 1024)
    
    # Line at 192KB - interpolate Z values along this Y
    x_line_192kb = np.linspace(X.min(), X.max(), 100)
    z_line_192kb = griddata((X, Y_log), Z, (x_line_192kb, np.full_like(x_line_192kb, y_log_192kb)), method='linear')
    
    fig.add_trace(go.Scatter3d(
        x=x_line_192kb,
        y=np.full_like(x_line_192kb, y_log_192kb),
        z=z_line_192kb,
        mode='lines',
        line=dict(color='red', width=5),
        name='L1/L2 Boundary (192KB)',
        legendrank=1,
        hovertemplate='L1/L2 Boundary<extra></extra>'
    ))
    
    # Line at 12MB - interpolate Z values along this Y
    x_line_12mb = np.linspace(X.min(), X.max(), 100)
    z_line_12mb = griddata((X, Y_log), Z, (x_line_12mb, np.full_like(x_line_12mb, y_log_12mb)), method='linear')
    
    fig.add_trace(go.Scatter3d(
        x=x_line_12mb,
        y=np.full_like(x_line_12mb, y_log_12mb),
        z=z_line_12mb,
        mode='lines',
        line=dict(color='red', width=5),
        name='L2/Memory Boundary (12MB)',
        legendrank=2,
        hovertemplate='L2/Memory Boundary<extra></extra>'
    ))
    
    # Create log2 size tick labels
    y_log_min, y_log_max = Y_log.min(), Y_log.max()
    y_ticks = np.arange(np.ceil(y_log_min), np.floor(y_log_max) + 1)
    y_labels = [naturalsize(2 ** int(tick), binary=True, format="%.0f") for tick in y_ticks]
    
    # Create stride axis ticks (showing multiples of 8 bytes)
    x_min, x_max = X.min(), X.max()
    x_tick_vals = np.arange(0, x_max + 16, 16)
    x_tick_labels = [str(int(val // 8)) for val in x_tick_vals]
    
    fig.update_layout(
        title='Visualizing spatial and temporal locality on an M1 chip',
        scene=dict(
            xaxis_title='Stride (×8 bytes)',
            yaxis_title='Size',
            xaxis=dict(
                tickvals=list(x_tick_vals),
                ticktext=x_tick_labels
            ),
            yaxis=dict(
                tickvals=list(y_ticks),
                ticktext=y_labels
            ),
            zaxis_title='Read Throughput (MB/s)',
            camera=dict(
                eye=dict(x=-1.5, y=1.5, z=0.2),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='cube'
        ),
        width=1200,
        height=600,
        hovermode='closest',
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Generate HTML with embedded text and GitHub link below
    html_content = fig.to_html(include_plotlyjs='cdn')
    
    # Append text content below the plot
    html_content += """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 40px auto; padding: 20px; line-height: 1.6; color: #333;">

        <p>Chapter 6.5 of <a href="https://books.google.co.uk/books/about/Computer_Systems.html?id=1SgrAAAAQBAJ&source=kp_book_description&redir_esc=y">Computer Systems: a Programmer's Perspective</a> introduces the concept of a memory mountain — a visualization of how memory read throughput varies with spatial and temporal locality. I constructed the chart by running this simple benchmark on my M1 Macbook Air: (also on <a href="https://github.com/kelvinou01/memory-mountain">GitHub</a>): </p>

        <pre style="background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; border-left: 4px solid #0366d6;">
<code>void test(int elems, int stride) {
    double result = 0.0;
    volatile double sink;
    for (int i = 0; i < elems; i += stride)
        result += data[i];
    sink = result;
}
</code></pre>

        <pre style="background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; border-left: 4px solid #0366d6;">
<code>double run(int size_bytes, int stride) {
    int elems = size_bytes / sizeof(double);
    if (elems > MAXELEMS) return -1;

    test(elems, stride);
    double t = seconds(test, elems, stride);
    if (t == 0) return -1;

    long reads = elems / stride;
    double mb = (reads * 8.0) / (1024.0 * 1024.0);
    return mb / t;
}
</code></pre>

        <p>We vary spatial locality by changing the stride, and temporal locality by changing the total size of all items copied. </p>

        <p>Looking at the graph, along the size axis, we see ridges — sudden changes in throughput — at the 192KB and 12MB marks. When the working set of the program increases beyond the M1's L1 data cache (192KB), we see sharp drops in performance as we switch to L2. Ditto from L2 (12MB) to main memory. </p>

        <p>Looking at the stride axis, when the entire working set fits in L2 memory, performance is constant, which I am guessing is due the CPU aggresively <a href="https://en.wikipedia.org/wiki/Prefetching">pre-fetching</a> based on the stride of previous accesses. When the working set can't fit into L2, we see a smooth drop in performance as worsening spatial locality decreases L2 hit rates. I have no idea what causes the huge fluctuations in performance when the entire working set fits into the L1 data cache. </p>

        </div>
    """
    
    with open(output, 'w') as f:
        f.write(html_content)
    
    print(f"Saved interactive 3D plot to {output}")

if __name__ == "__main__":
    app()