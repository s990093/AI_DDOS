import numpy as np
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text
from numba import jit

console = Console()

@jit(nopython=True)
def update_velocity(v, x, p, g, omega, phip, phig, rp, rg):
    """JIT-compiled velocity update function"""
    return omega * v + phip * rp * (p - x) + phig * rg * (g - x)

@jit(nopython=True)
def enforce_bounds(x, lb, ub):
    """JIT-compiled bounds enforcement"""
    x = np.maximum(x, lb)
    x = np.minimum(x, ub)
    return x


def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
        minstep=1e-8, minfunc=1e-8, debug=False):
    """
    Perform a particle swarm optimization (PSO) with Rich UI, Numba acceleration,
    and support for interruption with the current best result.
    """
    
    # Input validation
    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
    
    # Rich console setup
    console = Console()
    
    # Setup bounds
    vhigh = np.abs(ub - lb)
    vlow = -vhigh
    
    # Constraint handling
    obj = lambda x: func(x, *args, **kwargs)
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                console.print("[green]No constraints given.[/green]")
            cons = lambda x: np.array([0])
        else:
            if debug:
                console.print("[yellow]Converting ieqcons to a single constraint function[/yellow]")
            cons = lambda x: np.array([y(x, *args, **kwargs) for y in ieqcons])
    else:
        if debug:
            console.print("[yellow]Single constraint function given in f_ieqcons[/yellow]")
        cons = lambda x: np.array(f_ieqcons(x, *args, **kwargs))
        
    def is_feasible(x):
        check = np.all(cons(x)>=0)
        return check
        
    # Initialize the particle swarm
    S = swarmsize
    D = len(lb)
    x = np.random.rand(S, D)
    v = np.zeros_like(x)
    p = np.zeros_like(x)
    fp = np.zeros(S)
    g = []
    fg = 1e100
    
    # Rich progress display for initialization
    with console.status("[bold green]Initializing particle swarm...") as status:
        for i in range(S):
            x[i, :] = lb + x[i, :]*(ub - lb)
            p[i, :] = x[i, :]
            fp[i] = obj(p[i, :])
            
            if i==0:
                g = p[0, :].copy()
            
            if fp[i]<fg and is_feasible(p[i, :]):
                fg = fp[i]
                g = p[i, :].copy()
            
            v[i, :] = vlow + np.random.rand(D)*(vhigh - vlow)
    
    # Main iteration loop with Rich progress bar
    try:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("[cyan]Optimizing...", total=maxiter)
            
            it = 1
            while it <= maxiter:
                rp = np.random.uniform(size=(S, D))
                rg = np.random.uniform(size=(S, D))
                
                for i in range(S):
                    # Update velocity using Numba-accelerated function
                    v[i, :] = update_velocity(v[i, :], x[i, :], p[i, :], g, omega, phip, phig, rp[i, :], rg[i, :])
                    
                    # Update position and enforce bounds using Numba-accelerated function
                    x[i, :] = x[i, :] + v[i, :]
                    x[i, :] = enforce_bounds(x[i, :], lb, ub)
                    
                    fx = obj(x[i, :])
                    
                    if fx < fp[i] and is_feasible(x[i, :]):
                        p[i, :] = x[i, :].copy()
                        fp[i] = fx
                        
                        if fx < fg:
                            # if debug:
                            #     console.print(f"[bold green]Best after iteration {it} Value = {fg}[/bold green]")
                            
                            tmp = x[i, :].copy()
                            stepsize = np.sqrt(np.sum((g-tmp)**2))
                            
                            if np.abs(fg - fx) <= minfunc:
                                console.print(f"[bold green]Stopping search: Swarm best objective change less than {minfunc}[/bold green]")
                                return tmp, fx
                            elif stepsize <= minstep:
                                console.print(f"[bold green]Stopping search: Swarm best position change less than {minstep}[/bold green]")
                                return tmp, fx
                            else:
                                g = tmp.copy()
                                fg = fx
                
                # if debug:
                #     console.print(f"[blue]Best after iteration {it} Value = {fg}[/blue]")
                
                progress.update(task, advance=1)
                it += 1
    except KeyboardInterrupt:
        console.print("[red]Interrupted! Returning the current best solution...[/red]")
        return g, fg
    
    console.print("[yellow]Stopping search: maximum iterations reached --> {}[/yellow]".format(maxiter))
    
    if not is_feasible(g):
        console.print("[red]However, the optimization couldn't find a feasible design. Sorry[/red]")
    
    return g, fg
