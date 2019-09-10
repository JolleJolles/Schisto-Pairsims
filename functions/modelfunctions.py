import numpy as np
import torch as torch


class simmodel:
    def __init__(self,numparticles,numsteps,savestep=15):
        self.numparticles = numparticles
        self.beta = 1 # scaling to decrease turning amplitude with speed.  beta=1 means no change
        self.mu_s = np.ones(numparticles)

        self.tau_turn = np.ones(numparticles)*0.1
        self.tau_speed = np.ones(numparticles)
        self.sigma_speed = 0.2 # 0.2 was used in paper
        self.sigma_turn = 3 # 
        self.socialweight = 0.5  # 2 is approx the same as Ioaunnou model, but its too high for this - they always stay together then
        self.speedsocialweight = 0  # this defaults to zero
        self.mean_mu_s = 1  # should be 1.  This is used for setting the size of the repulsion, align, attract zones
        self.r_repulsion = 3.657*self.mean_mu_s
        self.r_align = 6.857*self.mean_mu_s
        self.r_attract = 20*self.mean_mu_s
        self.ignoresteps=[200,10**4]  # [how many steps to ignore social, how often to do it]
        self.maxturnangle = 10*(np.pi/180)
        self.p_turn_self = 0  # used for persistantzonal.  Persistantzonal with p_turn_self=0 is the same as classic zonal.  if p_turn_seflf=1, then will ignore social
        self.viewingzone=np.pi

        self.strictdirection = False

        # Stop-go parameters
        self.stopgosim=False
        self.numstates=2  # [go, stop]
        self.Tswitch = 10*np.ones((numparticles,self.numstates))
        self.statespeedmult = np.ones((numparticles,self.numstates))
        self.statespeedmult[:,1] = 0.2
        self.stopgosocial = 0.8
        self.sigma_stopgo=0.1        

        # Configuration
        self.numsimsteps=numsteps  #really, should do at least 10**6, probably more, to ensure sampling enough
        self.usevectorcalc = False
        self.savestep=savestep  # don't save all the simulation results - only save this many timesteps
        self.dt = 1/10
        self.xsize, self.ysize = [60,60]
        self.numsavesteps=np.floor(self.numsimsteps/self.savestep).astype(int)    
        self.periodic=True        


        # wall interaction functions, when use
        self.LJ_epsilon = 1/100; #amplitude
        self.LJ_sigma = .05 # distance
        self.LJ_wallcutoff=np.power(2,1/6)*self.LJ_sigma # keep only repulsive part of the LJ wall force
        self.wallturn_epsilon = 1/100;  # amplitude for wall turning epsilon
        self.wallturn_sigma = 0.05;  # distance for wall turning        
        
    def update_mean_mu_s(self,mean_mu_s):
        self.mean_mu_s = mean_mu_s  # should be 1.  This is used for setting the size of the repulsion, align, attract zones
        self.r_repulsion = 3.657*self.mean_mu_s
        self.r_align = 6.857*self.mean_mu_s
        self.r_attract = 20*self.mean_mu_s

    def getdiffs(self,ptcls):
        if (not self.periodic):
            print('Use periodic boundaries!  I will finish the boundary simulations later')
            return
        #  Now can assume that boundaries are periodic
        if self.usevectorcalc:
            return vectordiffs_periodic(ptcls,self.xsize,self.ysize)
        else:       
            return loopdiffs_periodic(ptcls,self.xsize,self.ysize)

    def getModelDirections(self,diff,angles):
        if self.speedsocialweight==0:  # then use the regular verion, without speed social changes
            cmspeed = np.zeros(self.numparticles)
            if self.usevectorcalc:
                if self.r_attract==self.r_align:
                    cmdirs = getCouzinModelDir_vector_torch(diff,angles,self.r_repulsion,self.r_align,self.viewingzone)
                else:
                    print('Vector calc is only written to work with single social zone')
                    return
            else:
                # not vector calc
                cmdirs = np.array([getCouzinModelDir(d,a,self.r_repulsion,self.r_align,self.r_attract,self.viewingzone) for d,a in zip(diff,angles)])
        else:  # use version with speed changes
            if self.usevectorcalc:
                print('Vector calc does not work with speed changes')
            else:
                cm=np.array([getCouzinModelDirSpeed(d,a,self.r_repulsion,self.r_align,self.r_attract,self.viewingzone) for d,a in zip(diff,angles)])
                cmdirs = cm[:,0:2]
                cmspeed = cm[:,2]                     
        return cmdirs, cmspeed


########################################################################################################################################
#### Distance functions
########################################################################################################################################
def vectordiffs_periodic(ptcls,xsize,ysize):
    # Use torch to do vector calculation distance, but just on the cpu
    numparticles = len(ptcls)
    positions = ptcls[:, 0:2]

    result = np.empty([numparticles, numparticles, 5])
    
    #create shifted positions
    # note that this will get the case wrong where a particle is simultaneously outside of "both" periodic boundaries.  But this is very rare and therefore should not be problem
    posup = np.copy(positions)
    posdown = np.copy(positions)
    posleft = np.copy(positions)
    posright = np.copy(positions)
    posleft[:,0] = posleft[:,0] - xsize
    posright[:,0] = posright[:,0] + xsize
    posup[:, 1] = posup[:, 1] + ysize
    posdown[:, 1] = posdown[:, 1] - ysize
    allpos = np.vstack(np.vstack([[positions], [posleft], [posright], [posup], [posdown]]).swapaxes(0,1))    
    #allpos = np.vstack((positions, posleft, posright, posup, posdown )) 

    #converts all numpy to tensors
    allpositions = torch.from_numpy(allpos)
    pos = torch.from_numpy(positions)

    #distance based calculations
    expanded_x1 = torch.unsqueeze(pos, 0)  
    expanded_x2 = torch.unsqueeze(allpositions, 1)
    #Distance between every pair of particles in x in every dimension (dx,dy)
    rx = expanded_x2 - expanded_x1
    diffs = rx
    # square distane for each particle pair in each dimension  (dx^2,dx^2)
    rx2 = rx.type(torch.FloatTensor) ** 2
    # absolute square distance between every pair of particles(dx^2+dx^2)
    r2 = torch.sum(rx2, 2)
    # absolute distance between every pair of particles
    dists = torch.sqrt(r2)

    #convert dists and diffs back into numpy 
    dists = dists.numpy()
    diffs = diffs.numpy()


    #creates velocity matrix
    velocity = ptcls[:, 2:4]
    vel = torch.from_numpy(velocity)
    expanded_v1 = torch.unsqueeze(vel, 0)  
    expanded_v2 = torch.unsqueeze(vel, 1)
    dv = expanded_v2 - expanded_v1
    vdiffs = dv.numpy()
    vdiffs = np.vstack(np.vstack([[vdiffs], [vdiffs], [vdiffs], [vdiffs], [vdiffs]]).swapaxes(0,1))   
    dvx = vdiffs[:, :, 0]
    dvy = vdiffs[:, :, 1]
    
    #masks and reshapes position difference and dist matrices 
    diffx=diffs[:,:,0]
    diffy=diffs[:,:,1]
    
    # make infinite distance for the particle with itself
#    lnp = np.arange(numparticles)
#    dists[lnp*5,lnp]=np.inf
    
    #concatenates and returns results as one array
#     result = np.concatenate((diffx, diffy, dists, vx, vy), 1)
    result= np.concatenate([[diffx],[diffy],[dists],[dvx],[dvy]],axis=0).T
    return result            

def loopdiffs_periodic(ptcls,xsize,ysize):
    numparticles=len(ptcls)
    pos=ptcls[:,0:2];
    vel=ptcls[:,2:4];
    result=np.empty([numparticles,numparticles,5]);
    for focus in range(numparticles):
        # get difference correct it for periodic boundaries
        diffs=pos-pos[focus];
        for q in range(numparticles):
            diffs[q]=correctdiff(diffs[q],xsize,ysize)
        # calculate distances
        dists=Vlen_array(diffs)
        # calculate velocity vector difference
        vdiffs=vel-vel[focus];
        # save the results
        result[focus]=np.concatenate((diffs,np.transpose([dists]),vdiffs),axis=1);        
        # returns:  [diff_x, diff_y, dist, diff_vx, diff_vy]
        #           [  0   ,   1   ,   2 ,   3,       4
    return result         
# helper function to correct for periodic boundaries, not using vector calc    
def correctdiff(dd,xsize,ysize): # Corrects the distance calculation for the periodic box size
    newdd=dd;
    if dd[0]<(-xsize/2): 
        newdd[0]=newdd[0]+xsize
    elif dd[0]>=(xsize/2): 
        newdd[0]=newdd[0]-xsize
    if dd[1]<(-ysize/2): 
        newdd[1]=newdd[1]+ysize
    elif dd[1]>=(ysize/2): 
        newdd[1]=newdd[1]-ysize      
    return newdd

########################################################################################################################################
# Model functions, to get zonal desired direction
########################################################################################################################################
def getCouzinModelDir(neighbors,focusangle, r_r, r_o, r_a, viewingzone):
    mixsocialzone = (r_a==r_o)
    repdir=np.array([0,0]);
    numrep=0;
    aligndir=np.array([0,0]);
    attractdir=np.array([0,0]);
    for nnum in range(len(neighbors)):
        # get the zone, and add to direction correction
        viewingangle=(focusangle-np.arctan2(neighbors[nnum,1],neighbors[nnum,0]))
        ndist=neighbors[nnum,2]
        if (np.cos(viewingangle)>np.cos(viewingzone)) & (ndist>0):
            if ndist<=r_r:
                #zone=1
                numrep=numrep+1
                repdir = repdir - neighbors[nnum,0:2]/ndist
            else:
                if ndist<=r_o:
                    #zone=2
                    # usee the difference in velocities, not the absolute velocity of the neighbor
                    aligndir = aligndir + neighbors[nnum,3:5]/Vlen(neighbors[nnum,3:5])
                        #zone=3
                if mixsocialzone:  # then just have a single "social" zone
                    if ndist<=r_a:
                        attractdir = attractdir + neighbors[nnum,0:2]/ndist
                else:
                    if (ndist<=r_a) & (ndist>r_o):
                        attractdir = attractdir + neighbors[nnum,0:2]/ndist
    # use the zonal directions to determine the new direction
    newdir=np.array([0,0]);
    if numrep>0:
        newdir = repdir;
    else:
        newdir = aligndir + attractdir;
    vlennewdir=Vlen(newdir);
    if vlennewdir>0:  # only normalize if >0, to avoid div by zero errors.  Also, it will return zero length vector, if no neighbors
        newdir=newdir/vlennewdir; 
    return newdir

def getCouzinModelDirSpeed(neighbors,focusangle,r_r, r_o, r_a, viewingzone,speedweights=[1,1]):
    mixsocialzone = (r_a==r_o)
    repdir=np.array([0,0]);
    numrep=0;
    aligndir=np.array([0,0]);
    attractdir=np.array([0,0]);
    repspeed = 0
    socialspeed = 0
    
    # rotated coordinates of neighbors:
    xrot = np.cos(-focusangle)
    yrot = np.sin(-focusangle)
    coords_rotated =np.dot([[xrot,-yrot],[yrot,xrot]],neighbors[:,0:2].T).T
    for nnum in range(len(neighbors)):
        # get the zone, and add to direction correction
        viewingangle=(focusangle-np.arctan2(neighbors[nnum,1],neighbors[nnum,0]))
        ndist=neighbors[nnum,2]
        
        if (np.cos(viewingangle)>np.cos(viewingzone)) & (ndist>0):
            if ndist<=r_r:
                #zone=1
                numrep=numrep+1
                repdir = repdir - neighbors[nnum,0:2]/ndist
                repspeed = repspeed - coords_rotated[nnum,0]
            else:
                dxrot = np.abs(coords_rotated[nnum,0])  # get abs value                
                if ndist<=r_o:  #zone=2
                    # use the difference in velocities, not the abs velocity of the neighbor
                    aligndir = aligndir + neighbors[nnum,3:5]/Vlen(neighbors[nnum,3:5])

                if mixsocialzone:  # then just have a single "social" zone
                    if ndist<=r_a:
                        attractdir = attractdir + neighbors[nnum,0:2]/ndist
                        if (dxrot>r_r) & (dxrot<=r_a):  # make speed changes if in 'social zone'
                            socialspeed = socialspeed + coords_rotated[nnum,0]                  
                else:
                    if (ndist<=r_a) & (ndist>r_o):
                        attractdir = attractdir + neighbors[nnum,0:2]/ndist
                        if (dxrot<=r_a) & (dxrot>r_o):  # make speed changes if in attract zone
                            socialspeed = socialspeed + coords_rotated[nnum,0]                                          
    # use the zonal directions to determine the new direction
    newdir=np.array([0,0]);
    dspeed = 0
    if numrep>0:
        newdir = repdir;
        dspeed = speedweights[0]*np.sign(repspeed)
    else:
        newdir = aligndir + attractdir;
        dspeed = speedweights[1]*np.sign(socialspeed)
    vlennewdir=Vlen(newdir);
    if vlennewdir>0:  # only normalize if >0, to avoid div by zero errors
        newdir=newdir/vlennewdir;    
    return np.append(newdir, dspeed)


def getCouzinModelDir_vector_torch(diff,angles,r_r,r_s,viewingzone):
    # this is a partially optimized way to calculate the desired direction.  It is "partially optimized", because some of the things below could probably be better - the baseline performance is about the same as the naiive cpu implementation.  However, with this, the scaling with larger number of neighbors is much better - adding a larger social zone (and therefore more neighbors), does not change the running time, whereas it does significantly for the regular version
    ntorch = torch.from_numpy(diff)
    viewtest = torch.from_numpy(np.cos([viewingzone]))
    anglestorch = torch.from_numpy(angles[:,None])

    if viewingzone < np.pi:
        viewneighbors = torch.cos(anglestorch-torch.atan2(ntorch[:,:,1],ntorch[:,:,0])) > viewtest
    else:
        viewneighbors = torch.ones(diff.shape[0:2])
        viewneighbors = viewneighbors.type(torch.uint8)

    # this wasn't any faster.
#     calc = torch.cos(anglestorch-torch.atan2(ntorch[:,:,1],ntorch[:,:,0]))
#     viewneighbors = calc.ge(viewtest)    

    repzone = viewneighbors & (ntorch[:,:,2]<= r_r) & (ntorch[:,:,2]>0) 
    socialzone = viewneighbors & (np.logical_not(repzone)) & (ntorch[:,:,2]<=r_s) & (ntorch[:,:,2]>0)

    userep = torch.sum(repzone,1)>0
    usesocial = np.logical_not(userep) & (torch.sum(socialzone,1)>0)
    
    repzonedouble = repzone.type(torch.DoubleTensor)
    socialzonedouble = socialzone.type(torch.DoubleTensor)
    
    xrep = ntorch[:,:,0]*repzonedouble
    yrep = ntorch[:,:,1]*repzonedouble
    distrep = ntorch[:,:,2][repzone]
    xrep[repzone] = xrep[repzone]/distrep
    yrep[repzone] = yrep[repzone]/distrep
    rx = - torch.sum(xrep,1)
    ry = - torch.sum(yrep,1)
    rx = rx[userep]
    ry = ry[userep]
    rnorm = torch.sqrt(rx**2+ry**2)
    rx, ry = rx/rnorm, ry/rnorm
    
    xsoc = ntorch[:,:,0]*socialzonedouble
    ysoc = ntorch[:,:,1]*socialzonedouble
    distsoc = ntorch[:,:,2][socialzone]  
    xsoc[socialzone] = xsoc[socialzone]/distsoc
    ysoc[socialzone] = ysoc[socialzone]/distsoc
    ax, ay = torch.sum(xsoc,1), torch.sum(ysoc,1)
    ax = ax[usesocial]
    ay = ay[usesocial]
    
    vxsoc = ntorch[:,:,3]*socialzonedouble
    vysoc = ntorch[:,:,4]*socialzonedouble
    vnorm = torch.sqrt(vxsoc[socialzone]**2+vysoc[socialzone]**2)
    vxsoc[socialzone] = vxsoc[socialzone]/vnorm
    vysoc[socialzone] = vysoc[socialzone]/vnorm
    ox, oy = torch.sum(vxsoc,1), torch.sum(vysoc,1)
    ox = ox[usesocial]
    oy = oy[usesocial]
    sx, sy = ax+ox, ay+oy
    snorm = torch.sqrt(sx**2+sy**2)
    sx = sx/snorm
    sy = sy/snorm
    
    newdirs = torch.zeros((len(diff),2))
    newdirs = newdirs.type(torch.DoubleTensor)
    newdirs[userep,0] = rx
    newdirs[userep,1] = ry
    newdirs[usesocial,0] = sx
    newdirs[usesocial,1] = sy    
    
    return newdirs.data.numpy()

########################################################################################################################################
# misc functions, used above to help with calculations
########################################################################################################################################

def fixanglerange(angles): # Puts all angles into a range of [-Pi,Pi] 
    return np.arctan2(np.sin(angles),np.cos(angles))
    
def Vlen(vec):  # vector length
    return np.sqrt(vec[0]*vec[0]+vec[1]*vec[1]) 

def Vlen_array(vecarray):
    return np.sqrt(vecarray[:,0]**2+vecarray[:,1]**2)
        




