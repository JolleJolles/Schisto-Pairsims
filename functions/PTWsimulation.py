import numpy as np
from IPython import display  # used for clearing plots, to view during the simulation

        
def ptwsimulation(simmodel, showprogress=False, simulator='ptw'):
    numparticles, beta, mu_s, tau_turn, tau_speed, sigma_speed, sigma_turn, socialweight = simmodel.numparticles, simmodel.beta, simmodel.mu_s, simmodel.tau_turn, simmodel.tau_speed, simmodel.sigma_speed, simmodel.sigma_turn, simmodel.socialweight
    mean_mu_s = simmodel.mean_mu_s
    
    numstates, stopgosim, statespeedmult, Tswitch, stopgosocial, sigma_stopgo = simmodel.numstates, simmodel.stopgosim, simmodel.statespeedmult, simmodel.Tswitch, simmodel.stopgosocial, simmodel.sigma_stopgo

    numsimsteps, usevectorcalc, savestep, dt, xsize, ysize, numsavesteps = simmodel.numsimsteps, simmodel.usevectorcalc, simmodel.savestep, simmodel.dt, simmodel.xsize, simmodel.ysize, simmodel.numsavesteps
    speedsocialweight = simmodel.speedsocialweight
    


    
    allparticles = np.zeros([numsavesteps, numparticles, 9])

    [startxpositions,startypositions]=[xsize*np.random.rand(numparticles),ysize*np.random.rand(numparticles)];
    startangles=2*np.pi*np.random.rand(numparticles)-np.pi;
    startspeeds=mu_s*np.ones([numparticles])

    startparticles = np.transpose([startxpositions, # x
                                   startypositions, # y
                                   startspeeds*np.cos(startangles), # vx
                                   startspeeds*np.sin(startangles), # vy
                                   startspeeds, # speed
                                   startangles, # orientation
                                   np.zeros([numparticles]), # angular velocity
                                   np.random.rand(numparticles), # stop-go accumulator variable
                                   np.random.randint(numstates,size=numparticles)  # 'state'
                                   ]);
    allparticles[0]=startparticles;

    # for optimizing running times and making code easier:
    ind_x, ind_y, ind_vx, ind_vy, ind_spd, ind_ang, ind_angvel, ind_stopgo, ind_state = np.arange(9)
    lnp = np.arange(numparticles)

    currentparticles=startparticles

    # this makes it so that the last entry of "allparticles" won't be zeros, if savestep evenly divides numsimsteps
    if savestep==1:
        numrun = numsimsteps
    else:
        numrun = numsimsteps+1

    step = 1
    ss = 0

    while ss<numsavesteps:
        
        # social interactions via zonal model
        diff=simmodel.getdiffs(currentparticles) 
        cmdirs, cmspeed = simmodel.getModelDirections(diff,currentparticles[:,ind_ang])

        # ignore social interactions if within the range to ignore:
        if (step>simmodel.ignoresteps[0]) & (np.mod(step,simmodel.ignoresteps[1])-simmodel.ignoresteps[0]<0):
            cmdirs = 0*cmdirs

        vxhat, vyhat = [np.cos(currentparticles[:,ind_ang]), np.sin(currentparticles[:,ind_ang])]  

        if simulator=='ptw':
            socialtorque = socialweight*(vxhat*cmdirs[:,1]-vyhat*cmdirs[:,0])
            # scale torque with speed
            betavals=beta_fn(currentparticles[:,ind_spd]/1,beta)  # leave mu_s here, i.e. don't make it zero
            # updates for angular velocity, orientation, speed
            angvel = currentparticles[:,ind_angvel] + dt/tau_turn*(betavals*socialtorque - currentparticles[:,ind_angvel]) + sigma_turn*np.sqrt(dt)*betavals*np.random.randn(numparticles)

        else:
            if simulator=='classiczonal':  # then use the "classic" Couzin model, where you specify the angle
                # check if the angle is too large
                zonalangle = np.arctan2(cmdirs[:,1],cmdirs[:,0])
                deltaangle = normAngle(zonalangle-currentparticles[:,ind_ang])

            elif simulator=='persistantzonal':  # then use the updated version from Ionnou, where there is persistance in preferred direction
                # do this calculation with vectors, because that is how the paper did it
                desired_vx = simmodel.p_turn_self*vxhat + (1-simmodel.p_turn_self)*cmdirs[:,0]
                desired_vy = simmodel.p_turn_self*vyhat + (1-simmodel.p_turn_self)*cmdirs[:,1]            
                zonalangle = np.arctan2(desired_vy,desired_vx)
                deltaangle = normAngle(zonalangle-currentparticles[:,ind_ang])

            clip = np.abs(deltaangle)>simmodel.maxturnangle
            deltaangle[clip] = np.sign(deltaangle[clip])*(simmodel.maxturnangle)[clip]
            #  add noise.  # the 3/2 comes from stepping the angle, instead of angular velocity
            deltaangle += sigma_turn*(dt**(3/2))*np.random.randn(numparticles)
            angvel = deltaangle/dt

            # so that they "find" each other, give ptw dynamics if social==zero
            notusingsocial = np.logical_not( (cmdirs[:,1]**2+cmdirs[:,0]**2) > 0 )
            angvel[notusingsocial] = currentparticles[notusingsocial,ind_angvel] + sigma_turn*np.sqrt(dt)*np.random.randn(np.sum(notusingsocial))


        # if is over.  do these below for all of the cases
        angle = normAngle(currentparticles[:,ind_ang] + dt*angvel)

        if simmodel.strictdirection==True:
            angle[0]=0
            angvel[0]=0
        
        # stop-go state matches
        if stopgosim:
            currentstates = (currentparticles[:,ind_state]).astype(int)
            statematches = np.array([s==currentstates for s in currentstates])*2-1  # +1 for match, -1 for not match
            insocialzone = diff[:,:,2]<simmodel.r_attract
            stateslope = np.array([(np.sum(st[z])-1)/np.sum(z) for st, z in zip(statematches,insocialzone)])
    #         stateslope = (np.sum(statematches,axis=1)-1)/(numparticles-1)  # mean after substract the state of self.  This has the mean going over all space, not just the social zone though
            stopgoaccum = currentparticles[:,ind_stopgo] + dt/Tswitch[lnp,currentstates]*(1-stopgosocial*stateslope) + sigma_stopgo*np.sqrt(dt)*np.random.randn(numparticles)
            stopgoaccum = np.maximum(stopgoaccum,0)
            # see if ones have crossed the threshold.  for these, reset them and switch the states
            toswitch = stopgoaccum>=1
            stopgoaccum[toswitch]=0
            states=currentstates.copy()   # new states
            states[toswitch] = np.mod(states[toswitch]+1,numstates)  # this 'cycles though' states, and could allow for more than 2 states
        else:
            currentstates = np.zeros(numparticles).astype(int)
            states = np.zeros(numparticles).astype(int)
            stopgoaccum = np.zeros(numparticles)

        speed = (currentparticles[:,ind_spd] + dt/tau_speed*(mu_s*statespeedmult[lnp,currentstates] + speedsocialweight*cmspeed - currentparticles[:,ind_spd]) 
                 + statespeedmult[lnp,currentstates]*sigma_speed*np.sqrt(dt)*np.random.randn(numparticles)/np.sqrt(tau_speed))
        speed = np.maximum(speed,0)

        # update the velocity
        vx = speed*np.cos(angle)
        vy = speed*np.sin(angle)
        xpos=np.mod(currentparticles[:,ind_x] + dt*vx,xsize)
        ypos=np.mod(currentparticles[:,ind_y] + dt*vy,ysize)
        newparticles = np.stack((xpos,ypos,vx,vy,speed,angle,angvel,stopgoaccum,states),axis=1)    

        # save for this step, if a savestep
        if np.mod(step,savestep)==0:
            allparticles[ss] = newparticles     

        currentparticles=newparticles        

        if np.mod(step,2000)==0:
            if showprogress:
                display.clear_output(wait=True)  
                print(step,numsimsteps)

        step = step+1
        ss=np.floor_divide(step,savestep)

    return allparticles



## calculate quantities for analysis
def dist_and_dcoords(allparticles,simmodel,showprogress=False):
    numsavesteps, numparticles, _ = allparticles.shape
    alldist = np.zeros([numsavesteps,numparticles,numparticles])
    alldcoords = np.zeros([numsavesteps,numparticles,numparticles,2])
    alldcoords_rotated = np.zeros([numsavesteps,numparticles,numparticles,2])
    ind_x, ind_y, ind_vx, ind_vy, ind_spd, ind_ang, ind_angvel = np.arange(7)
    def getmin(vals):
        return np.min(np.reshape(vals,(numparticles,numparticles,-1)),axis=2)
    for ss in range(numsavesteps):
        if np.mod(ss,2000)==0:
            if showprogress:
                display.clear_output(wait=True) 
                print(ss,numsavesteps)
        diff = simmodel.getdiffs(allparticles[ss])    
        dists=np.reshape(diff[:,:,2],(numparticles,numparticles,-1))
        alldist[ss] = np.min(dists,axis=2) 
        am = np.argmin(dists,axis=2) 
        # I can't think of a better way to do this, to reshape the arrays using vectors.  I tried for awhile... so, here, just have loops over j, which isn't needed, but does the job
        dx = np.reshape(diff[:,:,0],(numparticles,numparticles,-1))
        dy = np.reshape(diff[:,:,1],(numparticles,numparticles,-1))
        for i in range(numparticles):
            xrot = np.cos(-allparticles[ss,i,ind_ang])
            yrot = np.sin(-allparticles[ss,i,ind_ang])
            for j in range(numparticles):
                alldcoords[ss,i,j] = [ dx[i,j,am[i,j]] , dy[i,j,am[i,j]]]
            alldcoords_rotated[ss,i] = np.dot([[xrot,-yrot],[yrot,xrot]],alldcoords[ss,i].T).T
    return alldist, alldcoords_rotated

def beta_fn(spd_mu, beta,a=1):
    if beta==1:
        return np.ones(len(spd_mu))
    else:
        return a*np.power(beta,1-np.abs(spd_mu))
   
def normAngle(angles):
    return np.arctan2(np.sin(angles),np.cos(angles))

