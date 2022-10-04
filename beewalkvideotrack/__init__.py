import cv2
import numpy as np
import pandas as pd

class BeeTrack():
    def __init__(self,videofilename,fromframe=0,toframe=1000000000,downsample=10,thresholdfordetection=4.9,mmperpixel=None):
        self.videofilename = videofilename
        self.mmperpixel = mmperpixel
        self.fromframe = fromframe
        self.toframe = toframe
        self.downsample = downsample
        cap = cv2.VideoCapture(videofilename)
        self.framerate = cap.get(cv2.CAP_PROP_FPS)
        self.thresholdfordetection = thresholdfordetection
        
        allframes = []
        print("Loading frames")
        for i in range(toframe):
            print(".",end="")
            ret, temp = cap.read()
            if not ret: break
            if i<fromframe: continue #skip first 'fromframe' frames
            frame = np.mean(temp.astype(float)[::downsample,::downsample,:],2)
            allframes.append(frame)
            
        self.allframes = np.array(allframes)
        
        #normalise to mean of 80#15 and std of 27#9

        self.stdcorr = np.std(self.allframes)/27
        self.allframes/=self.stdcorr
        self.meancorr = np.mean(self.allframes)-80        
        self.allframes-=self.meancorr
        
    
    def particlefilter(self,frames,Nsamp = 10000,movescale=20,lookback=10,lookbackB=3):
        sh = frames[0].shape
        x = np.random.rand(Nsamp,2)*np.array(sh)
        path = []
        particlerecord = []
        for i in range(lookback):
            path.append([0,0,1000])
        for i in range(lookback,len(frames)-lookback):
            diff = (frames[i]-frames[i-lookback])
            diff_backB = (frames[i]-frames[i-lookbackB])
            diff_forward = (frames[i]-frames[i+lookback])
            
            idx = x.astype(int)
            logp = -13-diff[(idx[:,0],idx[:,1])]/36 #6
            
            logp += -26-diff_backB[(idx[:,0],idx[:,1])]/18 #6
            logp += -13-diff_forward[(idx[:,0],idx[:,1])]/36 #6
            

            #a = np.linspace(0,np.pi*2,10,endpoint=False)
            #xs = x+np.c_[np.cos(a)[:,None,None]*15,np.sin(a)[:,None,None]*15]
            #xs[xs<0]=0
            #xs[xs[:,:,0]>=sh[0],0]=sh[0]-1
            #xs[xs[:,:,1]>=sh[1],1]=sh[1]-1
            #idx = xs.astype(int)
            #logp += -10-np.min(diff[(idx[:,:,0],idx[:,:,1])],0)/2 #3

            #trying to downweight locations that aren't dark enough
            #frame = frames[i].copy()
            #frame[frame<-5]=-5
            #frame[frame>10]=10
            #logp -= frame[(idx[:,0],idx[:,1])]/10
            
            ps = np.exp(logp)
            ps /= np.sum(ps)
            newx = x[np.random.choice(Nsamp,Nsamp,replace=True,p=ps),:]
            newx = newx + np.random.randn(len(x),2)*movescale
            newx[newx[:,0]<0,0] = 0
            newx[newx[:,0]>=sh[0],0] = sh[0]-1
            newx[newx[:,1]<0,1] = 0
            newx[newx[:,1]>=sh[1],1] = sh[1]-1
            x = newx
            particlerecord.append(x[::100,:]) #just grab a handful
            path.append(np.r_[np.mean(x,0),np.mean(np.std(x,0))])
        for i in range(lookback):
            path.append([0,0,1000])
        path = np.array(path)    
        return path,np.array(particlerecord)
    
    def getpath(self,Nsamp = 10000,movescale=4):
        pforward,forwardrecord = self.particlefilter(self.allframes,Nsamp,movescale)#,lookback)
        pbackward,backwardrecord = self.particlefilter(self.allframes[::-1],Nsamp,movescale)#,lookback)
        pbackward = pbackward[::-1,:]
        backwardrecord = backwardrecord[::-1,:]    
        self.forwardrecord = forwardrecord
        self.backwardrecord = backwardrecord
        
        newcov = (pforward[:,2]**-2 + pbackward[:,2]**-2)**-1
        newmean = (pforward[:,0:2]/pforward[:,2:3]**2 + pbackward[:,0:2]/pbackward[:,2:3]**2)*newcov[:,None]
        newstd = np.sqrt(newcov)
        self.meanpath = newmean
        self.stdpath = newstd
        self.pforward = pforward
        self.pbackward = pbackward
        return newmean,newstd
    
    def moving_average(self,x, w=10):
        return np.convolve(x, np.ones(w), 'same') / w
    def moving_avg_path(self,path):
        return np.c_[self.moving_average(path[:,0]),self.moving_average(path[:,1])]

    def compute_features(self):
        path = self.moving_avg_path(self.meanpath)
        self.smoothpath = path
        speed = np.sum((np.diff(path,axis=0))**2,1)**.5 #pixels per frame
        speed = self.moving_average(speed,15)*self.downsample #we convert to original image size
        self.speed = speed
        self.walk = np.full(len(speed),np.NaN)
        keep = (self.stdpath<self.thresholdfordetection)[:-1]
        self.walk[keep & (self.speed<15)] = True
        self.walk[keep & (self.speed>25)] = False
        s = self.speed
        s[self.walk!=True]=False
        self.dist = np.cumsum(s)
        self.totalwalkdist = np.sum(s)
        
        #Compute segments walked
        walk = self.walk.copy()
        walk[-1]=np.NaN #ensures the walk stops at the end
        walkstarts = (walk[:-1]!=1) & (walk[1:]==1)
        walkstarts = np.where(walkstarts)[0]
        walkstops = (walk[:-1]==1) & (walk[1:]!=1)
        walkstops = np.where(walkstops)[0]

        walksegments = []
        for start,end in zip(walkstarts,walkstops):
            distwalked = np.sum(self.speed[start:end]) #in pixels
            if self.mmperpixel is None:
                mmdistwalked = None
            else:
                mmdistwalked = distwalked * self.mmperpixel #in mm
            walksegments.append([start,end,start/self.framerate,end/self.framerate,self.videofilename,distwalked,mmdistwalked])
        self.walksegments = pd.DataFrame(walksegments,columns = ['start (frame)','end (frame)','start (s)','end (s)','filename','distance (pixels)','distance (mm)'])
        
    def plotframe(self,i,clim=None,drawparticles=False,ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        ax.imshow(self.allframes[i],cmap='gray')
        if clim is not None: plt.clim(clim)
        statechar = '?'
        if self.walk[i]==True: statechar = 'W'
        if self.walk[i]==False: statechar = 'F'
        plt.title("Frame %4d, sd: %3.0f pixels. speed: %4.1f. dist: %4.1f %s" % (i,self.stdpath[i],self.speed[i],self.dist[i],statechar))
        if (i<len(self.stdpath)): 
            if (self.stdpath[i]<self.thresholdfordetection):
                if self.walk[i]==True:
                    plt.plot(self.smoothpath[i,1],self.smoothpath[i,0],'y+',markersize=60)
                #plt.plot(self.meanpath[i,1],self.meanpath[i,0],'y+',markersize=60)
                plt.gca().add_patch(plt.Circle((self.meanpath[i,1],self.meanpath[i,0]),2*self.stdpath[i], color='y', fill=False,clip_on=False))

        if (i<self.backwardrecord.shape[0]): 
            if drawparticles:
                plt.scatter(self.backwardrecord[i,:,1],self.backwardrecord[i,:,0],10,color='yellow')
                plt.scatter(self.forwardrecord[i,:,1],self.forwardrecord[i,:,0],10,color='green',marker='+')
                
        plt.xlim([0,self.allframes[0].shape[1]])
        plt.ylim([self.allframes[0].shape[0],0])

    def plotresult(self,fromframe=0,toframe=np.inf,step=10,clim=None,drawparticles=False,ax=None):
        """
        Plot a subset of the frames, with the particle estimated path of the bee.

        fromframe = start frame, default=0
        toframe =  end frame, default=last frame
        step = how many frames to step forward between images, default 10
        clim = plot scale limits (default to the full range of the image)
        E.g. plotresult(130,150,1,clim=[-10,10]) plots frames 130,131,...149.
             And scales the grey image values to show those from -10 to +10.
        """
        import matplotlib.pyplot as plt
        toframe = min(len(self.allframes),toframe)
        for i in range(fromframe,toframe,step):  
            plt.figure(figsize=[20,10])
            self.plotframe(i,clim,drawparticles,ax)

    def grabhighres(self,box=100,downsample=1,skip=1,runfn=None):
        self.highresdownsample = downsample
        cap = cv2.VideoCapture(self.videofilename)
        self.highresframes = []
        self.runresults = []
        self.highresframeindices = []
        print("Loading frames")
        #framecache = []
        i=0
        for fileidx in range(self.toframe):
            print(".",end="")
            ret, temp = cap.read()
            if not ret: break
            if fileidx<self.fromframe: continue #skip first 'fromframe' frames

            #temp = np.mean(temp.astype(float),2)
            #framecache.append(temp)
            #if len(framecache)>5: framecache.pop(0)
            if (self.stdpath[i]<self.thresholdfordetection) and (fileidx%skip==0):
                pos = (self.meanpath[i,:]*self.downsample).astype(int)
                temp = np.mean(temp.astype(float),2)

                img = temp#-framecache[0]
                frame = img[pos[0]-box:pos[0]+box:downsample,pos[1]-box:pos[1]+box:downsample]
                frame/=self.stdcorr
                frame-=self.meancorr
                
            else:
                frame = None
            if runfn is not None:
                self.runresults.append(runfn(frame))
            else:
                self.highresframes.append(frame)
            self.highresframeindices.append([fileidx,i])
            i+=1
            
    def gettracksummarydataframe(self,fromframe=0,toframe=np.inf,step=1):
        import pandas as pd            
        results = []
        toframe = min(len(self.allframes),toframe)
        for i in range(fromframe,toframe,step):
            if (self.stdpath[i]<self.thresholdfordetection):
                rec = [i,self.meanpath[i,1],self.meanpath[i,0],self.walk[i],self.speed[i],self.stdpath[i]]
            else:
                rec = [i,None,None,None,None,self.stdpath[i]]
            results.append(rec)
        results = np.array(results)
        return pd.DataFrame(results[:,1:],columns=['x','y','walking','speed','error'],index=results[:,0])

    
    def makemovie(self,filename,clim=None,drawparticles=False):
        from moviepy.editor import VideoClip
        from moviepy.video.io.bindings import mplfig_to_npimage
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        def make_frame(t):
            ax.clear()
            self.plotframe(int(t*25),clim=clim,drawparticles=drawparticles,ax=ax)    
            return mplfig_to_npimage(fig)

        animation = VideoClip(make_frame, duration = (len(self.allframes)-1)/25)

        #animation.ipython_display(fps = 25, loop = False, autoplay = True)
        animation.write_videofile(filename,fps=25)
