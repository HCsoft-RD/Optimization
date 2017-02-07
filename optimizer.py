import time#
import pandas as pd
import numpy as np
import random
import copy

from scipy.optimize import  differential_evolution,minimize
from scipy.optimize import basinhopping
from scipy.optimize import  differential_evolution

from IPython.core.display import clear_output

from shaolin.core.dashboard import Dashboard
from test_functions import TestFunctions

class Optimizer(Dashboard):
    """Widget interface to scipy global optimization algorithms.
    It allows to select the parameters for a global optimizer.
    
    Supported algorithms are Basin hopping and Diferential evolution. 
    This class executes the solver on a function object.
    
    Attributes
    ----------
    algos : list, ['differential_evolution','basin_hopping']
        Contains the names of the available solvers.
    epoch: int
        Current iteration of the selected solver.
    max_epoch: int
        Maximum number of iterations of the selected solver.
    max_reads: int
        Maximum number of functions reads allowed for the selected solver.
    pot_func: ObjectiveFunction object.
        Object representing the objective function to be solved. 
    best: (aray,float),  (n_components,),1
        tuple containing the minimum value found for the objective function
        and the parameters that yielded that value.
    """
    def __init__(self,objfunc,select='Basin hopping',fractal_kwargs={},basinhopping_kwargs={},diffevo_kwargs={},**kwargs):
        
        differential_evolution = DifferentialEvolution(objfunc,name='differential_evolution',**diffevo_kwargs)
        basin_hopping = BasinHopping(objfunc,name='basin_hopping',**basinhopping_kwargs)

        
        dash = ["c$N=optimizer_dash",[differential_evolution,
                                      basin_hopping,
                                      ["r$n=btn_row",
                                        ["togs$N=algo_sel&o=['Differential evolution','Basin hopping']&val="+str(select),
                                         'btn$d=Run&n=run_btn']
                                      ]
                                     ]
               ]
        
        Dashboard.__init__(self,dash,**kwargs)
        self.algos = ['differential_evolution','basin_hopping']
        self._last_eval = select.lower().replace(' ','_')
        self.algo_sel.observe(self.update_layout)
        self.run_btn.observe(self.run)
        self.update_layout()
    
    def run(self,_=None, name=None, end_callback=None):
        """Solves the objective funtion with the currently selected parameters"""
        opt = self.algo_sel.value.lower().replace(' ','_') if name is None else name
        self._last_eval = opt
        getattr(self,opt).run(end_callback=end_callback)
    
    @property 
    def epoch(self):
        return getattr(self,self._last_eval).epoch
    @epoch.setter
    def epoch(self,val):
        getattr(self,self._last_eval).epoch = val
        
    @property 
    def max_epoch(self):
        return getattr(self,self._last_eval).max_epoch
    @max_epoch.setter
    def max_epoch(self,val):
        getattr(self,self._last_eval).max_epoch = val
        
    @property 
    def max_reads(self):
        return getattr(self,self._last_eval).max_reads
    @max_reads.setter
    def max_reads(self,val):
        getattr(self,self._last_eval).max_reads = val
    
    @property
    def pot_func(self):
        return getattr(self,self._last_eval).pot_func
    
    @pot_func.setter
    def pot_func(self,val):
        getattr(self,self._last_eval).pot_func = val
    
    @property
    def best(self):
        optimizer = getattr(self,self._last_eval)
        return (optimizer.best_pos_so_far, float(optimizer.best_so_far))
    
    def reset(self):
        """Reset the solvers"""
        self.basin_hopping.reset()
        self.differential_evolution.reset()
    
    def update_layout(self,_=None):
        """widget interface updating"""
        if self.algo_sel.value == "Differential evolution":
            self.differential_evolution.visible = True
            self.differential_evolution.run_btn.visible = False
            self.basin_hopping.visible = False
          
        elif self.algo_sel.value == "Basin hopping":
            self.differential_evolution.visible = False
            self.basin_hopping.visible = True
            self.basin_hopping.run_btn.visible = False


class DifferentialEvolution(Dashboard):
    """Shaolin interface for the Differential Evolution solver from the scipy package.
    Parameters
    ----------
    objfunc: ObjectiveFunction object.
        Object representing the objective function to be solved. 
    #This is from scipy.optimize.differential_evolution
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
        The default is 'best1bin'.
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize * len(x)`` individuals.
    tol : float, optional
        When the mean of the population energies, multiplied by tol,
        divided by the standard deviation of the population energies
        is greater than 1 the solving process terminates:
        ``convergence = mean(pop) * tol / stdev(pop) > 1``
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        ``U[min, max)``. Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : int or `np.random.RandomState`, optional
        If `seed` is not specified the `np.RandomState` singleton is used.
        If `seed` is an int, a new `np.random.RandomState` instance is used,
        seeded with seed.
        If `seed` is already a `np.random.RandomState instance`, then that
        `np.random.RandomState` instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Display status messages
    callback : callable, `callback(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the population convergence.  When ``val`` is greater than one
        the function halts. If callback returns `True`, then the minimization
        is halted (any polishing is still carried out).
    polish : bool, optional
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
        method is used to polish the best population member at the end, which
        can improve the minimization slightly.
    init : string, optional
        Specify how the population initialization is performed. Should be
        one of:
            - 'latinhypercube'
            - 'random'
        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space. 'random' initializes
        the population randomly - this has the drawback that clustering can
        occur, preventing the whole of parameter space being covered.
    
    Attributes
    ----------
    epoch: int
        Current iteration of the selected solver.
    pot_func: ObjectiveFunction object.
        Object representing the objective function to be solved. 
    best: (aray,float),  (n_components,),1
        Tuple containing the minimum value found for the objective function
        and the parameters that yielded that value.
    best_pos_so_far: (array), (n_components)
        Alias for the best parameters found by the optimizer.
    best_so_far: float
        Alias for minimum value found by the optimizer.
    """
    def __init__(self,
                 objfunc,
                 strategy='best1bin',
                 maxiter=1000,
                 popsize=15,
                 max_reads=150000,
                 tol=0.01,
                 mutation=(0.5, 1),
                 recombination=0.7,
                 seed=160290,
                 callback=None,
                 disp=False,
                 polish=True,
                 init='latinhypercube',
                 **kwargs):
        
        def false():
            return False
        self.pot_func = objfunc
        self.best_pos_so_far = None
        self.best_so_far = 1e20
        self._ext_callback =lambda x,conv: False if callback is None else callback
        self._run_callback = false
        self.epoch=0
        strats = ['best1bin','best1exp','rand1exp','randtobest1exp','best2exp','rand2exp','randtobest1bin','best2bin','rand2bin','rand1bin']
        dash = ['r$N=differential_evolution',["##Differential evolution$n=title",
                                              ['r$N=controls_row',[["c$N=first_row",["(0,100000,1,"+str(maxiter)+")$d=Maxiter&n=max_epoch",
                                                               "(0,1e8,1,"+str(max_reads)+")$d=Max reads",
                                                               "(1,1e5,1,"+str(int(popsize))+")$d=Popsize"]],
                                             
                                              ["c$N=sec_row",["(0.,10.,.01,"+str(tol)+")$d=Tol",
                                                               "(0.,100.,0.1,"+str(mutation)+")$d=Mutation",
                                                               "(0,1e8,1,"+str(recombination)+")$d=Recombination"]
                                              ],
                                            ["c$N=third_row",["(0,1000000,1,"+str(seed)+")$d=seed",
                                                               "dd$d=Strategy&o="+str(strats)+"&val="+str(strategy),
                                                               "togs$d=Init&o=['latinhypercube','random']",
                                                               ]
                                              ],["c$n=btn_col",["["+str(bool(polish))+"]$d=Polish","btn$d=Run&n=run_btn"]]]
               ]]]
        Dashboard.__init__(self,dash,**kwargs)
        self.run_btn.observe(self.run)
        
    @property
    def best(self):
        return (self.best_pos_so_far.copy(), float(self.best_so_far))
        
    
    def run(self,_=None, end_callback=None):
        """Solves the objective funtion"""
        self.reset()
        if not end_callback is None:
            self._run_callback = end_callback
        result = differential_evolution(self.pot_func.evaluate,
                                        self.pot_func.domain,
                                        callback=self.callback,
                                        maxiter=int(self.max_epoch.value),
                                        popsize=int(self.popsize.value),
                                        tol=self.tol.value,
                                        mutation=self.mutation.value,
                                        recombination=self.recombination.value,
                                        strategy=self.strategy.value,
                                        init=self.init.value,
                                        polish=self.polish.value)
        
        clear_output(True)
        print(result)
        
        self.update_best(result.x)

    def callback(self,x,convergence):
        """Callback support for custom stopping of the function"""
        self.update_best(x)
        self.epoch += 1
        end = self._ext_callback(x,convergence)

        return end or self._run_callback() 
    
    def reset(self):
        """Resets the solver attributes"""
        self.best_pos_so_far = None
        self.best_so_far = 1e20
        self.epoch=0
        self.pot_func.n_reads = 0
    
    def update_best(self,x):
        """Keeps track of the best value and parameters found by the solver"""
        val = self.pot_func.evaluate(x)
        if self.best_so_far>val:
            self.best_pos_so_far = x
            self.best_so_far = val

class MyBounds(object):
    """Class in charge of managing the boundaries of the sampling region for the basin hopping solver."""
    def __init__(self, xmax=[1.1,1.1], xmin=[-1.1,-1.1],other_test=None ):
        def true(**kwargs):
            return True
        self._outer_accept_test = true if other_test is None else other_test
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        out = self._outer_accept_test(**kwargs)
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin and out

class BasinHopping(Dashboard):
    """Find the global minimum of a function using the basin-hopping algorithm
    
    Parameters
    ----------
    objfunc: ObjectiveFunction object.
        Object representing the objective function to be solved. 
    x0 : ndarray
        Initial guess.
    niter : integer, optional
        The number of basin hopping iterations
    T : float, optional
        The "temperature" parameter for the accept or reject criterion.  Higher
        "temperatures" mean that larger jumps in function value will be
        accepted.  For best results ``T`` should be comparable to the
        separation
        (in function value) between local minima.
    stepsize : float, optional
        initial step size for use in the random displacement.
    minimizer_kwargs : dict, optional
        Extra keyword arguments to be passed to the minimizer
        ``scipy.optimize.minimize()`` Some important options could be:
            method : str
                The minimization method (e.g. ``"L-BFGS-B"``)
            args : tuple
                Extra arguments passed to the objective function (``func``) and
                its derivatives (Jacobian, Hessian).
    take_step : callable ``take_step(x)``, optional
        Replace the default step taking routine with this routine.  The default
        step taking routine is a random displacement of the coordinates, but
        other step taking algorithms may be better for some systems.
        ``take_step`` can optionally have the attribute ``take_step.stepsize``.
        If this attribute exists, then ``basinhopping`` will adjust
        ``take_step.stepsize`` in order to try to optimize the global minimum
        search.
    accept_test : callable, ``accept_test(f_new=f_new, x_new=x_new, f_old=fold, x_old=x_old)``, optional
        Define a test which will be used to judge whether or not to accept the
        step.  This will be used in addition to the Metropolis test based on
        "temperature" ``T``.  The acceptable return values are True,
        False, or ``"force accept"``. If any of the tests return False
        then the step is rejected. If the latter, then this will override any
        other tests in order to accept the step. This can be used, for example,
        to forcefully escape from a local minimum that ``basinhopping`` is
        trapped in.
    callback : callable, ``callback(x, f, accept)``, optional
        A callback function which will be called for all minima found.  ``x``
        and ``f`` are the coordinates and function value of the trial minimum,
        and ``accept`` is whether or not that minimum was accepted.  This can be
        used, for example, to save the lowest N minima found.  Also,
        ``callback`` can be used to specify a user defined stop criterion by
        optionally returning True to stop the ``basinhopping`` routine.
    interval : integer, optional
        interval for how often to update the ``stepsize``
    disp : bool, optional
        Set to True to print status messages
    niter_success : integer, optional
        Stop the run if the global minimum candidate remains the same for this
        number of iterations.
        
        
    Attributes
    ----------
    epoch: int
        Current iteration of the selected solver.
    pot_func: ObjectiveFunction object.
        Object representing the objective function to be solved. 
    best: (aray,float),  (n_components,),1
        Tuple containing the minimum value found for the objective function
        and the parameters that yielded that value.
    best_pos_so_far: (array), (n_components)
        Alias for the best parameters found by the optimizer.
    best_so_far: float
        Alias for minimum value found by the optimizer.
        """
    def __init__(self,
                 objfunc,
                 x0=None,
                 max_reads=10000000,
                 niter=100,
                 T=1.0,
                 stepsize=0.5,
                 minimizer_kwargs=None,
                 take_step=None,
                 accept_test=None,
                 callback=None,
                 interval=50,
                 disp=False,
                 niter_success=None,
                 **kwargs):
        def false(**kwargs):
            return False
        def true(**kwargs):
            return True
        self._outer_accept_test = true if accept_test is None else accept_test
        self.minimizer_kwargs = minimizer_kwargs
        self.take_step = take_step
        
       
        self.pot_func = objfunc
        self.x0 = x0
        self.best_pos_so_far = None
        self.best_so_far = 1e20
        self._ext_callback =lambda x,val,accept: False if callback is None else callback
        self._run_callback = false
        self.epoch=0
        niter_success = 0 if niter_success is None else niter_success
        strats = ['best1bin','best1exp','rand1exp','randtobest1exp','best2exp','rand2exp','randtobest1bin','best2bin','rand2bin','rand1bin']
        dash = ['r$N=differential_evolution',["##Basin-hopping$n=title",
                                              ['r$N=controls_row',[["c$N=first_row",["(0,100000,1,"+str(niter)+")$d=Niter&n=max_epoch",
                                                               "(0,1e8,1,"+str(max_reads)+")$d=Max reads",
                                                               "(0,1e4,1,"+str(int(niter_success))+")$d=niter_success"]],
                                             
                                              ["c$N=sec_row",["(0.,10000.,.01,"+str(T)+")$d=T",
                                                              "(0,1e8,1,"+str(interval)+")$d=Interval",
                                                               "(0.,100.,0.01,"+str(stepsize)+")$d=Stepsize"]
                                              ],
                                              ["c$n=btn_col",["["+str(bool(disp))+"]$d=Disp","btn$d=Run&n=run_btn"]]]
               ]]]
        Dashboard.__init__(self,dash,**kwargs)
        self.run_btn.observe(self.run)
    
    
    def accept_test(self,**kwargs):
        """Custom acceptance test support"""
        out = self._outer_accept_test(**kwargs)
        return out or self.pot_func.in_domain(kwargs['x'])
    
    
    
    @property
    def best(self):
        return (self.best_pos_so_far.copy(), float(self.best_so_far))
    
        
    
    def run(self,_=None, end_callback=None):
        """Solves the objective function"""
        self.reset()
        if not end_callback is None:
            self._run_callback = end_callback
        niter_success = None if self.niter_success.value == 0 else int(self.niter_success.value)
        x0 = self.pot_func.random_in_domain()
        
        xmax = [x[1] for x in self.pot_func.domain]
        xmin = [x[0] for x in self.pot_func.domain]
        
        bounds = MyBounds(xmax=xmax,xmin=xmin,other_test=self._outer_accept_test)
        result = basinhopping(self.pot_func.evaluate,
                              x0,
                              niter=self.max_epoch.value,
                              T=self.t.value,
                              stepsize=self.stepsize.value,
                              minimizer_kwargs=self.minimizer_kwargs,
                              take_step=self.take_step,
                              accept_test=bounds,
                              callback=self.callback,
                              interval=self.interval.value,
                              disp=self.disp.value,
                              niter_success=niter_success)
        
        clear_output(True)
        print(result)
        self.update_best(result.x)

    def callback(self,x,val,accept):
        """custom callback for finishing the optimization in a scipy-like manner."""
        self.update_best(x)
        self.epoch += 1
        end = self._ext_callback(x,val,accept)
        return end or self._run_callback() 
    
    def reset(self):
        """Resets the solver attributes"""
        self.best_pos_so_far = None
        self.best_so_far = 1e20
        self.epoch=0
        self.pot_func.n_reads = 0
    
    def update_best(self,x):
        """Keeps track of the best value and parameters found by the solver"""
        val = self.pot_func.evaluate(x)
        if self.best_so_far>val:
            self.best_pos_so_far = x
            self.best_so_far = val