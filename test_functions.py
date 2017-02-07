import numpy as np
from shaolin.core.dashboard import Dashboard
class ObjFunc(object):
    def __init__(self, func,domain,benchmark=None):
        self.func=func
        self.domain=[d[0] for d in domain]
        self.benchmark=benchmark
        self.n_reads = 0
        max_dom = []
        min_dom = []
        for pair in self.domain:
            #for pair in self.domain[dim]:
            max_dom.append(pair[1])
            min_dom.append(pair[0])
        self.max_dom = np.array(max_dom)
        self.min_dom = np.array(min_dom)
        
    def to_scaled(self,x):
        OldMax=self.max_dom
        OldMin = self.min_dom
        NewMax = 1.
        NewMin = -1.
        return (((x - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    
    def unscale(self,x):
        OldMax = 1.
        OldMin = -1.
        NewMax = self.max_dom
        NewMin = self.min_dom
        return (((x - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
       
       
    
    def evaluate(self,x):
        if len(x.shape)==1:
            #x = self.unscale(x_scaled)
            self.n_reads += 1
            return self.func(np.tile(x,(2,1)))[0]
        self.n_reads += len(x[:,0])
        #x = self.unscale(x_scaled)
        return self.func(x)
         
    
    def in_domain(self,x,scaled=False):
        if scaled:
            return np.all(np.abs(x)<=1)
        else:
            #for dim in range(len(self.domain)):
            for i,pair in enumerate(self.domain):
                    if x[i]<pair[0]:
                        return False
                    if x[i]>pair[1]:
                        return False 
            return True
    
    def random_in_domain(self):
            Xrd=[] 
            for pair in self.domain:   
                
                bot=pair[0]
                top=pair[1]
                Xrd.append(np.random.uniform(low=bot,high=top))
            return np.array(Xrd)
        
def eggholder(x):
    return -1*(x[:,1]+47)*np.sin(np.sqrt(np.abs(x[:,1]+x[:,0]/2.0+47.0)))-x[:,0]*np.sin(np.sqrt(np.abs(x[:,0]-(x[:,1]+47.0))))



class Rastrigin(ObjFunc):
    
    def __init__(self,n_dims=2,A=10):
        self.n_dims = n_dims
        dom = [[(-5.12,5.12)] for _ in range(self.n_dims)]
        bench=[np.zeros(n_dims),0]
        #fun = lambda X: -1.*(A*self.n_dims+np.sum([xi**2-A*np.cos(2*np.pi*xi) for xi in X]))
        fun = lambda x: A*self.n_dims+(x**2-A*np.cos(2*np.pi*x)).sum(axis=1)
        ObjFunc.__init__(self,fun,dom,bench)

class Rosenbrock(ObjFunc):
    
    def __init__(self,n_dims=2,A=10):
        self.n_dims = n_dims
        dom = [[(-50,50)] for _ in range(self.n_dims)]
        bench=[np.ones(n_dims),0.]
        #fun = lambda X: -1.*(A*self.n_dims+np.sum([xi**2-A*np.cos(2*np.pi*xi) for xi in X]))
        fun = lambda x: np.array([100*(x[:,i+1]-x[:,i]**2)**2+(x[:,i]-1)**2 for i in range(n_dims-1)]).sum(axis=0)
        ObjFunc.__init__(self,fun,dom,bench)

class StyblinskiTang(ObjFunc):
    
    def __init__(self,n_dims=2,A=10):
        self.n_dims = n_dims
        dom = [[(-5,5)] for _ in range(self.n_dims)]
        bench=[np.ones(n_dims)*-2.903534,39.16616*n_dims]
        #fun = lambda X: -1.*(A*self.n_dims+np.sum([xi**2-A*np.cos(2*np.pi*xi) for xi in X]))
        fun = lambda x: (x**4-16*x**2+5*x).sum(axis=1)/2.
        ObjFunc.__init__(self,fun,dom,bench)
        

class DeVilliersGlasser02(ObjFunc):
    def __init__(self):
        dom = [[(1,60)] for _ in range(5)]
        bench=[np.array([53.81,1.27,3.012,2.13,0.507]),0.]
        ObjFunc.__init__(self,self.calc_obj_func,dom,bench)
    
    def calc_obj_func(self,x):
        val = 0
        for i in range(1,24):
            ti = 0.1*(i-1)
            yi = 53.81*(1.27**ti)*np.tanh(3.012*ti+np.sin(2.13*ti))*np.cos(np.exp(0.507)*ti)
            val += (x[:,0]*x[:,1]**ti*np.tanh(x[:,2]*ti+np.sin(x[:,3]*ti))*np.cos(ti*np.exp(x[:,4]))-yi)**2
        return val

class LennardJones(ObjFunc):
    
    def __init__(self, n_atoms=10):
        self.dom = [-1.1,1.1]
        domain = [[(-1.1,1.1)] for _ in range(3*n_atoms)]
        self.N = n_atoms
        minima = {'2':-1,'3':-3,'4':-6,'5':-9.103852,'6':-12.712062,'7':-16.505384,
                  '8':-19.821489,'9':-24.113360,'10':-28.422532,'11':-32.765970,'12':-37.967600,
                  '13':-44.326801,'14':-47.845157,'15':-52.322627
                 }
        bench = [np.zeros(self.N*3),minima[str(n_atoms)]]
        ObjFunc.__init__(self,self.lj_func,domain,bench)
        
    def lj_func(self,x):
        def lennard_jones(U):
            U = U.reshape(self.N,3)
            npart = len(U)
            Epot = 0.0
            for i in range(npart):
                for j in range(npart):
                    if i>j:
                        r2 = np.linalg.norm(U[j,:]-U[i,:])**2
                        r2i = 1.0/r2
                        r6i = r2i*r2i*r2i
                        Epot = Epot + r6i*(r6i-1.)
            Epot = Epot * 4
            return Epot
        return np.array([lennard_jones(x[i,:]) for i in range(x.shape[0])]).reshape(x.shape[0],1)
    
    def random_in_domain(self):
        return self.to_scaled(np.random.uniform(low=self.dom[0],high=self.dom[1],size=(self.N,3)).flatten())
    
    def in_domain(self,x):
        return np.all(np.abs(x)<1.)

class MultiDimTest(Dashboard):
    
    def __init__(self,
                 ros_step=1,
                 ras_step=1,
                 sbt_step=1,
                 
                 lj_step=1,
                 ras_range=(2,4),
                 ros_range=(2,4),
                 lj_range=(2,4),
                 sbt_range=(2,4),
                 **kwargs):
        self.functions = {}
        
        dash = ['r$N=multim_test',[["c$n=ras_col",["[False]$d=Rastrigin&n=ras_tog",
                                                   "(2,500,1,"+str(ras_range)+")$d=Dim range&n=ras_dim",
                                                   "(1,99,1,"+str(ras_step)+")$d=Dim range&n=ras_step",
                                                  ]
                                 ],
                                 ["c$n=ros_col",["[False]$d=Rosenbrock&n=ros_tog",
                                                 "(2,500,1,"+str(ras_range)+")$d=Dim range&n=ros_dim",
                                                 "(1,99,1,"+str(ros_step)+")$d=Dim range&n=ros_step",
                                                 ]
                                 ],
                                 ["c$n=sbt_col",["[False]$d=StyblinskiTang&n=sbt_tog",
                                                 "(2,500,1,"+str(ras_range)+")$d=Dim range&n=sbt_dim",
                                                 "(1,99,1,"+str(ros_step)+")$d=Dim range&n=sbt_step",
                                                 ]
                                 ],
                                 ["c$n=lj_col",["[False]$d=LennardJones&n=lj_tog",
                                                "(2,500,1,"+str(ras_range)+")$d=Dim range&n=lj_dim",
                                                 "(1,99,1,"+str(ros_step)+")$d=Dim range&n=lj_step",
                                                 ]
                                 ],"[True]$d=DeVilliersGlasser02&n=dv2_tog",
                                   "btn$d=Update&n=update_btn"
                                ]
             ]
    
        Dashboard.__init__(self,dash,**kwargs)
        self.update_btn.observe(self.update)
        self.update()
        
    def update(self,_=None):
        funcs = {}
        if self.ras_tog.value:
            for i in range(self.ras_dim.value[0],self.ras_dim.value[1],self.ras_step.value):
                name = "rastrigin_"+str(i)
                funcs[name] = Rastrigin(i)
        if self.ros_tog.value:
            for i in range(self.ros_dim.value[0],self.ros_dim.value[1],self.ros_step.value):
                name = "rosenbrock_"+str(i)
                funcs[name] = Rosenbrock(i)
        if self.sbt_tog.value:
            for i in range(self.sbt_dim.value[0],self.sbt_dim.value[1],self.sbt_step.value):
                name = "styblinski_tang_"+str(i)
                funcs[name] = StyblinskiTang(i)
        if self.lj_tog.value:
            for i in range(self.lj_dim.value[0],self.lj_dim.value[1],self.lj_step.value):
                name = "lennard_jones_"+str(i)
                funcs[name] = LennardJones(i)
        if self.dv2_tog.value:
            funcs['dev_glass02'] = DeVilliersGlasser02()
        self.functions = funcs

class Wikipedia2D(Dashboard):
    
    
    def __init__(self,**kwargs):
        self.init_funcs()
        self.functions = {}
        dash = ["@selmul$d=Wikipedia test 2D&n=func_sel&o="+str(list(self._all_funcs.keys()))]
        
        Dashboard.__init__(self,dash,**kwargs)
        self.func_sel.value = tuple(self._all_funcs.keys())
        self.observe(self.update)
        self.update()
    
    def init_funcs(self):
        funcs = {}
        #Eggholder
        eggholder = lambda x: -1*(x[:,1]+47)*np.sin(np.sqrt(np.abs(x[:,1]+x[:,0]/2.0+47.0)))-x[:,0]*np.sin(np.sqrt(np.abs(x[:,0]-(x[:,1]+47.0))))
        dom=[[(-510,512)],[(-512,512)]]
        egg_bench=[np.array([512,404.231805]),-959.640663]
        egg=ObjFunc(eggholder,domain=dom,benchmark=egg_bench)
        funcs['eggholder'] = egg
        #Rastriguin
        funcs['rastrigin'] = Rastrigin(n_dims=2)
        #Ackley
        ackley = lambda x: -20*np.exp(-0.2*np.sqrt(0.5*(x[:,0]**2+x[:,1]**2))) - np.exp(0.5*((np.cos(2*np.pi*x[:,0])+np.cos(2*np.pi*x[:,1]))))+np.e+20
        ack_dom=[[(-5,5)],[(-5,5)]]
        ack_bench=[np.array([0.,0.]),0.]
        funcs['ackley'] = ObjFunc(ackley,domain=ack_dom,benchmark=ack_bench)
        #sphere
        sp = lambda x: (x**2).sum(axis=1)                                                                            
        sp_dom=[[(-5000,5000)],[(-5000,5000)]]
        sp_bench=[np.array([0.,0.]),0.]
        funcs['sphere'] = ObjFunc(sp,domain=sp_dom,benchmark=sp_bench)
        #rosenbrock
        funcs['rosenbrock'] = Rosenbrock(n_dims=2)  
        #Beale's
        beales = lambda x: (1.5-x[:,0]+x[:,0]*x[:,1])**2+(2.25-x[:,0]+x[:,0]*x[:,1]**2)**2+(2.625-x[:,0]+x[:,0]*x[:,1]**3)**2
        bea_dom=[[(-4.5,4.5)],[(-4.5,4.5)]]
        bea_bench=[np.array([3.,0.5]),0.]       
        funcs['beales'] = ObjFunc(beales,domain=bea_dom,benchmark=bea_bench)        
        self._all_funcs = funcs
        #Goldstein price
        gpf = lambda x: (1+(x[:,0]+x[:,1]+1)**2*(19-14*x[:,0]+3*x[:,0]**2-14*x[:,1]+6*x[:,0]*x[:,1]+3*x[:,1]**2))*\
                (30+(2*x[:,0]-3*x[:,1])**2*(18-32*x[:,0]+12*x[:,0]**2+48*x[:,1]-36*x[:,0]*x[:,1]+27*x[:,1]**2))
        gpf_dom =[[(-2,2)],[(-2,2)]]
        gpf_bench=[np.array([0.,-1]),3]
        funcs['goldstein_price'] = ObjFunc(gpf,domain=gpf_dom,benchmark=gpf_bench)
        #booth's function
        bth = lambda x: (x[:,0]+2*x[:,1]-7)**2+(2*x[:,0]+x[:,1]-5)**2
        bth_dom =[[(-10,10)],[(-10,10)]]
        bth_bench=[np.array([1,3.]),0]
        funcs['booth'] = ObjFunc(bth,domain=bth_dom,benchmark=bth_bench)
        #bukin6
        bk6 = lambda x: 100*np.sqrt(np.abs(x[:,1]-0.01*x[:,0]**2))+0.01*np.abs(x[:,0]+10)
        bk6_dom = [[(-15,-5)],[(-3,3)]]
        bk6_bench = [np.array([-10,1]),0.]
        funcs['bukin6'] = ObjFunc(bk6,domain=bk6_dom,benchmark=bk6_bench)
        #matyas
        mat = lambda x: 0.26*(x[:,0]**2+x[:,1]**2)-0.48*x[:,0]*x[:,1]
        mat_dom = [[(-10,10)],[(-10,10)]]
        mat_bench = [np.array([0,0]),0.]
        funcs['matyas'] = ObjFunc(mat,domain=mat_dom,benchmark=mat_bench)
        #levy13
        l13 = lambda x: np.sin(3*np.pi*x[:,0])**2+(x[:,0]-1)**2*(1+np.sin(3*np.pi*x[:,1])**2)+(x[:,1]-1)**2*(1+np.sin(2*np.pi*x[:,1])**2)
        l13_dom = [[(-10,10)],[(-10,10)]]
        l13_bench = [np.array([1,1]),0.]
        funcs['levy13'] = ObjFunc(l13,domain=l13_dom,benchmark=l13_bench)
        #Three-hump camel function
        thc = lambda x: 2*x[:,0]**2-1.05*x[:,0]**4+(x[:,0]**6)/6+x[:,0]*x[:,1]+x[:,1]**2
        thc_dom = [[(-5,5)],[(-5,5)]]
        thc_bench = [np.array([0,0]),0.]
        funcs['three_hump_camel'] = ObjFunc(thc,domain=thc_dom,benchmark=thc_bench)
        #easom function
        eas = lambda x: -np.cos(x[:,0])*np.cos(x[:,1])*np.exp(-((x[:,0]-np.pi)**2+(x[:,1]-np.pi)**2))
        eas_dom = [[(-100,100)],[(-100,100)]]
        eas_bench = [np.array([np.pi,np.pi]),-1.]
        funcs['easom'] = ObjFunc(eas,domain=eas_dom,benchmark=eas_bench)
        #McCornick
        mck = lambda x: np.sin(x[:,0]+x[:,1])+(x[:,0]-x[:,1])**2-1.5*x[:,0]+2.5*x[:,1]+1
        mck_dom = [[(-1.5,4)],[(-3,4)]]
        mck_bench = [np.array([-0.5471972,-1.5471975]),-1.913223]
        funcs['mccornick'] = ObjFunc(mck,domain=mck_dom,benchmark=mck_bench)
        #shaffer2
        sh2 = lambda x: 0.5+(np.sin(x[:,0]**2-x[:,1]**2)**2-0.5)/(1+0.001*(x[:,0]**2+x[:,1]**2))**2
        sh2_dom = [[(-100,100)],[(-100,100)]]
        sh2_bench = [np.array([0,0]),0.]
        funcs['shaffer2'] = ObjFunc(sh2,domain=sh2_dom,benchmark=sh2_bench)
        #shaffer4
        sh4 = lambda x: 0.5+(np.cos(np.sin(np.abs(x[:,0]**2-x[:,1]**2)))**2-0.5)/(1+0.001*(x[:,0]**2+x[:,1]**2))**2
        sh4_dom = [[(-100,100)],[(-100,100)]]
        sh4_bench = [np.array([0,1.25313]),0.292579]
        funcs['shaffer4'] = ObjFunc(sh4,domain=sh4_dom,benchmark=sh4_bench)
        
    def update(self,_=None):
        funcs = {}
        for key in self.func_sel.value:
            funcs[key]  =  self._all_funcs[key]
        self.functions = dict(funcs)                                                                             

class TestFunctions(Dashboard):
    def __init__(self,ros_step=1,
                 ras_step=1,
                 sbt_step=1,
                 lj_step=1,
                 ras_range=(2,4),
                 ros_range=(2,4),
                 lj_range=(2,4),
                 sbt_range=(2,4),
                 **kwargs):
        multi = MultiDimTest(ros_step=ros_step,
                                  ras_step=ras_step,
                                  sbt_step=sbt_step,
                                  lj_step=lj_step,
                                  ras_range=ras_range,
                                  ros_range=ros_range,
                                  lj_range=lj_range,
                                  sbt_range=sbt_range,
                                  name="multi"
                             )
        wiki = Wikipedia2D(name='wiki')
        self.functions = {}
        dash = ['c$N=test_functions',[multi,wiki,["r$n=btn_col",["togs$N=mode_sel&o=['wiki','multi']"]]]]
        Dashboard.__init__(self,dash,**kwargs)
        self.mode_sel.observe(self.update_layout)
        self.wiki.func_sel.observe(self.update)
        self.multi.update_btn.observe(self.update)
        self.update_layout()
        self.update()
    def update_layout(self,_=None):
        if self.mode_sel.value == 'wiki':
            self.wiki.visible=True
            self.multi.visible=False

        elif self.mode_sel.value == 'multi':
            self.wiki.visible=False
            self.multi.visible=True
            
    def update(self,_=None):
        #self.wiki.update()
        #self.multi.update()
        funcs = dict(self.wiki.functions)
        funcs.update(dict(self.multi.functions))
        self.functions = funcs