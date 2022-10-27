import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d #needed for getting size of tiles etc.
import os
from skimage.feature import match_template

#note about Katy's data. 10 tiles [40mm] = 357 pixels across, 13 pixels down. Each tile is 4mm across. So sqrt(357^2+13^2)=357.24=357..
#40 / 357 = 0.11205 mmperpixel

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)
    
class BeeTrack():
    def	 __init__(self,videofilename,fromframe=0,toframe=None,smoothtime=None,boundingbox=None,targetresolution=384,blur=None,thresholdfordetection=7,mmperpixel=None,resample='avg',framesplitcount=1800,unsmoothedspeedwalkthreshold=19.5*0.111):
        """
        Tracks a bee in 'top down' videos, produced by Katy Chapman at Exeter University.
        
        Parameters:
            videofilename = the filename of the video to process [required]
            fromframe,toframe = the frames to start and finish processing on.
            smoothtime,blur = the smoothtime runs a rectangular convolution over time, 'blurring' frames together
                              blur runs a Gaussian convolution over space 'blurring' the images
                              Both parameters default to None, but to use set smoothtime to an integer number of frames to combine,
                              and/or set blur to a foating point scale parameter of the Gaussian smoothing kernel.
            boundingbox = default None, otherwise set to [top left, bottom right] coordinates. [x1,y1,x2,y2]
            targetresolution = the resolution to run the algorithm at (default = 384 pixels wide)
            thresholdfordetection=7
            mmperpixel=None,
            framesplitcount=1800,
            unsmoothedspeedwalkthreshold, thresholdfordetection = These control when the bee is decided to be walking and whether a bee
                              is detected at all. unsmoothedspeedwalkthreshold sets a speed of 2.16mm/frame as the unsmoothed speed the
                              bee can be travelling before it is consider to be flying. The standard deviation in the particle model must
                              be less than 7 to be considered to have found the bee.
        resample = {'avg', 'quick'}: how to resample the raw video (whether to average each square or just sample every
                              nth pixel (quick). default 'avg' (this is particularly necessary if the video is noisy)        
        """
        if not os.path.exists(videofilename):
            print("File not found")
            assert False, "File not found" #need to replace with exception
        if toframe is None:
            toframe = fromframe+framesplitcount

        self.unsmoothedspeedwalkthreshold = unsmoothedspeedwalkthreshold
        cap = cv2.VideoCapture(videofilename)
        self.originalresolution = np.array([int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))])
        downsample = int(self.originalresolution[0]/targetresolution)
        print("Downsampling by %d" % downsample)
        self.videofilename = videofilename
        self.fromframe = fromframe
        self.toframe = toframe
        self.downsample = downsample
        
        self.framerate = cap.get(cv2.CAP_PROP_FPS)
        self.thresholdfordetection = thresholdfordetection
        
        allframes = []

        print("Loading frames")
        self.firstframeavg = None #we compute the average of the first 40 frames for getting the scale...
        firstframecount = 0
        self.frame_number = toframe
        
        for i in range(toframe):
            print(".",end="",flush=True)
            ret, temp = cap.read()
            if not ret: 
                self.frame_number = None
                break
            temp = np.array(temp).astype(float)
            temp = np.mean(temp.astype(float),2)
            if boundingbox is not None:
                temp = temp[boundingbox[0]:boundingbox[2],boundingbox[1]:boundingbox[3]]
            if i==fromframe: 
                newframeshape = (np.array(temp.shape)/downsample).astype(int)
                self.first_fromframe = temp.copy()  
                self.rawmeanval = np.mean(temp.flatten())
                self.rawstdval = np.std(temp.flatten())
                #if blur is not None: self.prep_blur(newframeshape,blur)
            if i<fromframe: continue #skip first 'fromframe' frames
            if i<fromframe+40: 
                if self.firstframeavg is None:
                    self.firstframeavg=temp.copy()
                else:
                    self.firstframeavg+=temp
                firstframecount+=1
            if resample=='avg':
                if i==0: print("Using 'averaging' downsampling")
                
                frame = rebin(temp,newframeshape)
            else:
                frame = temp[::downsample,::downsample]
            if blur is not None:
                frame = gaussian_filter(frame, sigma=blur)
                #frame = self.blur_image(frame)
            allframes.append(frame)
        self.allframes = np.array(allframes)            
        if smoothtime is not None:
            from scipy import signal
            self.allframes = signal.convolve(self.allframes, np.ones([smoothtime,1,1])/smoothtime,mode='same')

        self.firstframeavg /= firstframecount

        
        #normalise to mean of 80 and std of 27
        #normalises standard deviation and mean PER FRAME
        self.stdcorr = np.std(self.allframes,(1,2))[:,None,None]/27
        self.allframes/=self.stdcorr
        self.meancorr = np.mean(self.allframes,(1,2))[:,None,None]-80    
        self.allframes-=self.meancorr
        
        if mmperpixel is not None:
            self.mmperpixel = mmperpixel
            print("User Specified %0.4fmm/pixel." % self.mmperpixel)
        else:
            self.mmperpixel = self.estimate_scale()
            print("Estimating %0.4fmm/pixel." % self.mmperpixel)
            
    def resize(self,im,r):
        """
        Rescales by averaging blocks of pixels, returns lower res image. Scaled by r.
        """
        return im[np.arange(0,im.shape[0],r).astype(int),:][:,np.arange(0,im.shape[1],r).astype(int)]

    def estimate_scale(self):
        """
        We try to estimate the mm per pixel using a chunk of the image, in which we know the scale.
        This chunk of image is scaled and cross-correlated with the average of the first few frames.
        The scale that has the highest score is chosen, and the relative mm per pixel returned.
         
        The chunk of image was collected with:    
            bt1 = BeeTrack('/home/mike/Documents/Research/beebehaviourdata/labelled/19N_e_vert-R52.mp4',fromframe=10,toframe=50)
            search = bt1.firstframeavg[800:1400,1500:2100].copy()
            np.save('searchpattern',search)
        """    
        print("Estimating image scale...")
        searchpatfilename = os.path.join(os.path.dirname(__file__), 'searchpattern.npy')
        search = np.load(searchpatfilename) #the search pattern has 600/16.825 pixels per square (each square is 4mm x 4mm)
        
        maxr = None
        maxv = 0
        
        s = self.firstframeavg.shape
        s = [int(0.14*s[0]),int(0.83*s[0]),int(0.31*s[1]),int(0.73*s[1])]
        chunk = self.firstframeavg[s[0]:s[1]:self.downsample,s[2]:s[3]:self.downsample]
        
       
        for r in np.arange(self.downsample*0.2,self.downsample*4,0.05):
            tempsearch = self.resize(search,r)
            tempsearch-=np.mean(tempsearch)
            tempsearch/=np.std(tempsearch)
            
            try:
                maxconv = np.max(match_template(chunk,tempsearch))         
            except ValueError:
                continue
            if maxconv > maxv:
                maxv = maxconv
                maxr = r
        #So the image chunk is scaled by downsample
        #The search tile is scaled by maxr
        #so we find the ratio - which gives us how much we need to make our search smaller to match the image
        maxr = maxr/self.downsample        
       
        #maxr is therefore number of times smaller the images is than the search pattern...
        return 0.111*maxr #returns mmperpixel
        
        
    def particlefilter(self,frames,Nsamp = 50000,movescale=20,lookback=10,lookbackB=3):
        """
        This runs a particle filter (in one direction) over the frames of the video.
        
        Parameters:
          frames = list of 2d greyscale video frames, each with mean 80, std 27.
          Nsamps = number of particles [default is 50,000, but override by call from getpath, which defaults to 10,000]
          movescale = how fast they can diffuse [default is 20, but override by call from getpath, which defaults to 4pixels/frame]
          lookback, lookbackB = when computing differences between frames, how far to look back (and forward).
        Returns a tuple:
          path = an N by 4 array of mean and standard deviations of the particle locations each frame.
          particlerecord = subset of particles for debugging/plotting
          
        Description:
        Each particle's log probability of being on the bee, is computed by finding the value of the particle in the current 
        frame and subtracting it from (a) frame i-lookback, (b) i-lookbackB, (c) i+lookback.
        [lookback default = 10, lookbackB default = 3]
        The three differences are scaled added, and a constant added. The probability is computed, by exponenting (and normalised).
        
        The particle are then selected with replacement using this probability distribution.
        There is random Gaussian noise added with standard deviation 'movescale' [default = 20 pixels/frame]
        If the particle leaves the frame, it is moved back into the frame's boundary.
        """
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
        """
        Parameters:
            Nsamp = Number of particles (default 10000).
            movescale = how fast they can diffuse (default 4).
        Returns a tuple:
            - an N by 2 array of mean locations each frame.
            - an N array of standard deviations each frame.
        Key values stored in object:
            self.meanpath = the location of the bee (each frame)
            self.stdpath = the uncertainty in the path (each frame)
        Description:
            Uses the particle filter forward and backward to do smoothing: Takes the mean and standard deviation
            of the forward and backward filters. At each frame it combines the two distributions. It just uses
            one dimension of the standard deviation to do this. TO DO: Might want to improve this in future.      
        """
        print("Processing %d frames." % len(self.allframes))
        pforward,forwardrecord = self.particlefilter(self.allframes,Nsamp,movescale)#,lookback)
        pbackward,backwardrecord = self.particlefilter(self.allframes[::-1],Nsamp,movescale)#,lookback)
        if len(backwardrecord)==0:
            return None, None
            
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
        """
        Convolves x with a rectangular kernel (normalised). Returns result.
        """
        return np.convolve(x, np.ones(w), 'same') / w
    def moving_avg_path(self,path):
        """
        Convolves first two columns of x with a rectangular kernel (normalised). Returns result as an Nx2 array.
        """
        return np.c_[self.moving_average(path[:,0]),self.moving_average(path[:,1])]

    def compute_features(self):
        """
        Computes key features, using the results of the call to getpath.
        Parameters: None.
        Returns: Nothing.
        
        Description:
          - computes a smoothed path
          - computes the speed from both the smoothed path and the unsmoothed path
          - the unsmoothed path speed is smoothed though.
          - self.walk array is set to False or True if bee is found or None if not.
          - speeds converted to mm/frame.
             - Assumed to be found if stdpath (the standard deviation of the particle smoother) is less than thresholdfordetection
                    (set previously to, for example 7 pixels).
             - Walking if unsmoothedspeed<unsmoothedspeedwalkthreshold (set previously to 2.16mm/frame) and the smoothed speed is less than 
                     2.775 mm/frame.  
          - Saves key results in object:
             - self.totalwalkdist and self.totalwalkdistmm are both the same, in mm
             - self.dist in mm (cumulative sum)
             - self.walksegments is a dataframe, each row says: ['start (frame)','end (frame)','start (s)','end (s)','filename','distance (pixels)','distance (mm)','mm per pixel'].

        """
        path = self.moving_avg_path(self.meanpath)
        self.smoothpath = path
        speed = np.sum((np.diff(path,axis=0))**2,1)**.5 #pixels per frame
        speed = self.moving_average(speed,15)*self.downsample*self.mmperpixel #we convert to original image size
        unsmoothedspeed = np.sum((np.diff(self.meanpath,axis=0))**2,1)**0.5
        unsmoothedspeed = self.moving_average(unsmoothedspeed,15)*self.downsample*self.mmperpixel
        #unsmoothedspeed *= self.downsample
        self.speed = speed
        self.unsmoothedspeed = unsmoothedspeed
        self.walk = np.full(len(speed),np.NaN)
        keep = (self.stdpath<self.thresholdfordetection)[:-1]
        #self.walk[keep & (self.speed<15*0.111)] = True
        #self.walk[keep & (self.speed>25*0.111)] = False
        
        self.walk[keep & (self.unsmoothedspeed<self.unsmoothedspeedwalkthreshold)] = True
        self.walk[keep & (self.speed>25*0.111)] = False
        
        s = self.speed
        s[self.walk!=True]=False
        self.dist = np.cumsum(s)
        self.totalwalkdist = np.sum(s)
        #if self.mmperpixel is None:
        #    self.totalwalkdistmm = None
        #else:
        self.totalwalkdistmm = self.totalwalkdist#*self.mmperpixel
        
        #Compute segments walked
        walk = self.walk.copy()
        walk[-1]=np.NaN #ensures the walk stops at the end
        walkstarts = (walk[:-1]!=1) & (walk[1:]==1)
        walkstarts = np.where(walkstarts)[0]
        walkstops = (walk[:-1]==1) & (walk[1:]!=1)
        walkstops = np.where(walkstops)[0]
        walksegments = []
        for start,end in zip(walkstarts+1,walkstops+1):
            distwalked = np.sum(self.speed[start:end]) #in pixels
            if self.mmperpixel is None:
                mmdistwalked = None
            else:
                mmdistwalked = distwalked# * self.mmperpixel #in mm
            walksegments.append([start+self.fromframe,end+self.fromframe,(start+self.fromframe)/self.framerate,(end+self.fromframe)/self.framerate,self.videofilename,distwalked,mmdistwalked,self.mmperpixel])
        self.walksegments = pd.DataFrame(walksegments,columns = ['start (frame)','end (frame)','start (s)','end (s)','filename','distance (pixels)','distance (mm)','mm per pixel'])
        
    def plotframe(self,i,clim=None,drawparticles=False,ax=None):
        """
        Plots a single frame.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        ax.imshow(self.allframes[i],cmap='gray')
        if clim is not None: plt.clim(clim)
        statechar = '?'
        if self.walk[i]==True: statechar = 'W'
        if self.walk[i]==False: statechar = 'F'
        plt.title("Frame %4d, sd: %3.0f pixels. speed: %4.1f [%4.1f]. dist: %4.1f %s" % (i,self.stdpath[i],self.speed[i],self.unsmoothedspeed[i],self.dist[i],statechar))
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
        """
        Takes small hi-res images of the bee (in frames where it is found). 
        Stores in self.highresframes.
        """
        self.highresdownsample = downsample
        cap = cv2.VideoCapture(self.videofilename)
        self.highresframes = []
        self.runresults = []
        self.highresframeindices = []
        print("Loading frames")
        #framecache = []
        i=0
        for fileidx in range(self.toframe):
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
                frame/=np.std(frame)/27
                frame-=np.mean(frame)-80 
                
            else:
                frame = None
            if runfn is not None:
                self.runresults.append(runfn(frame))
            else:
                self.highresframes.append(frame)
            self.highresframeindices.append([fileidx,i])
            i+=1
            
    def gettracksummarydataframe(self,fromframe=0,toframe=np.inf,step=1):
        """
        Returns a dataframe holding track of the bee, whether it was felt the bee was walking, its speed and the standard deviation in the particle smoother's estimate.
        """
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
        """
        Generates a diagnostic/debug movie, saved in 'filename'.
        """
        from moviepy.editor import VideoClip
        from moviepy.video.io.bindings import mplfig_to_npimage
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        def make_frame(t):
            ax.clear()
            self.plotframe(int(t*25),clim=clim,drawparticles=drawparticles,ax=ax)    
            return mplfig_to_npimage(fig)

        animation = VideoClip(make_frame, duration = (len(self.allframes)-2)/25)

        #animation.ipython_display(fps = 25, loop = False, autoplay = True)
        animation.write_videofile(filename,fps=25)
