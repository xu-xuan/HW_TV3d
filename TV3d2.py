import numpy as np

	
def worker(In, J, dt, lam):
  ep = 1e-4
  m,n,l = J.shape
  for i in range(8):
    start = int(i*m/8-1)
    end = int((i+1)*m/8+1)
    #print(start,end)	
    if start < 0:
      start = start +1
      DfJx=J[(start+1):(end+1),:,:]-J[start:end,:,:]	  
    elif end > m:
      end = end-1
      DfJx=J[list(range(start+1,end))+[end-1],:,:]-J[start:end,:,:]
    else:
      DfJx=J[(start+1):(end+1),:,:]-J[start:end,:,:]
    DfJy=J[start:end,list(range(1,n))+[n-1],:]-J[start:end,:,:]
    DfJz=J[start:end,:,list(range(1,l))+[l-1]]-J[start:end,:,:]
    TempDJ=(ep+DfJx*DfJx+DfJy*DfJy+DfJz*DfJz)**(1/2)
    DivJx=DfJx/TempDJ;
    DivJy=DfJy/TempDJ;
    DivJz=DfJz/TempDJ;  
    del TempDJ
    mi,ni,li = DivJx.shape
    if start==0:
      div = DivJx[0:(mi-1),:,:]-DivJx[[0]+list(range(mi-2)),:,:] \
        + DivJy[0:(mi-1),:,:]-DivJy[0:(mi-1),[0]+list(range(ni-1)),:]\
        + DivJz[0:(mi-1),:,:]-DivJz[0:(mi-1),:,[0]+list(range(li-1))]
      J[start:(end-1),:,:] += dt * div -dt*lam*(J[start:(end-1),:,:]-In[start:(end-1),:,:])  
    elif end == m:
      mi = mi + 1
      end = end + 1	  
     # print(mi)
      div = DivJx[1:(mi-1),:,:]-DivJx[0:(mi-2),:,:] \
        + DivJy[1:(mi-1),:,:]-DivJy[1:(mi-1),[0]+list(range(ni-1)),:]\
        + DivJz[1:(mi-1),:,:]-DivJz[1:(mi-1),:,[0]+list(range(li-1))]
      J[(start+1):(end-1),:,:] += dt * div -dt*lam*(J[(start+1):(end-1),:,:]-In[(start+1):(end-1),:,:])  
    else:
      div = DivJx[1:(mi-1),:,:]-DivJx[0:(mi-2),:,:] \
        + DivJy[1:(mi-1),:,:]-DivJy[1:(mi-1),[0]+list(range(ni-1)),:]\
        + DivJz[1:(mi-1),:,:]-DivJz[1:(mi-1),:,[0]+list(range(li-1))]
      J[(start+1):(end-1),:,:] += dt * div -dt*lam*(J[(start+1):(end-1),:,:]-In[(start+1):(end-1),:,:])  	
  return J

	
if __name__ == '__main__':
  from mpi4py import MPI
  import matplotlib.pyplot as plt # plt 用于显示图片
  T, dt, lam = 200, 0.1, 0.01 
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  # np = 2, ncoord = [2,1,1]
  # The initial distribution
  if rank == 0:
    nx, ny , nz = 200, 200, 200
	# Generate image
    img = 100.0*np.ones((nx,ny,nz))
    img[75:150,75:150,75:150] = 150.0
	# Adding Gaussian noise
    nmean, nsigma = 0.0, 12.0
    nimg = np.random.normal(nmean,nsigma,(nx,ny,nz)) + img
    # plt.figure()
    # plt.imshow(img[:,100,:],"gray")
    # plt.savefig('./img.png')
    # plt.figure()
    # plt.imshow(nimg[:,100,:],"gray")
    # plt.savefig('./nimg.png')	
    del img
    nimgsile = nimg[:,:,0:101]
    sendbuf = nimg[:,:,99:200].copy()
    # plt.figure()
    # plt.imshow(sendbuf[:,100,:],"gray")
    # plt.savefig('./sendbuf.png')	
    comm.Send(sendbuf, dest=1, tag=11)
    # plt.figure()
    # plt.imshow(nimgsile[:,100,:],"gray")
    # plt.savefig('./nimgsile0.png')	
    del nimg, sendbuf
  else: 
    nimgsile = np.empty([200,200,101],dtype=np.float64) 
    comm.Recv(nimgsile, source=0, tag=11)
    plt.figure()
    plt.imshow(nimgsile[:,100,:],"gray")
    plt.savefig('./nimgsile1.png')	

  # Iterative
  J = nimgsile.copy()
  n0,n1,n2 = J.shape  
  for t in range(T):
    if rank ==0 and not t%5:
      print(t, 'of ', T)
    J = worker(nimgsile, J, dt, lam)
    #print('good work', rank )	
    if rank == 0:
      sendbuf = J[:,:,n2-2].copy()
      #print('begin send of', rank )	  
      comm.Send(sendbuf, dest=1, tag=100)
      del sendbuf
      recbuf = np.empty(J[:,:,n2-1].shape, dtype=J[:,:,n2-1].dtype)	  
      #print('goodsend of', rank )
      comm.Recv(recbuf, source=1, tag=110)	
      #print('goodrecv of', rank )
      J[:,:,n2-1] = recbuf
      del recbuf
    else:
      recbuf = np.empty(J[:,:,n2-1].shape, dtype=J[:,:,n2-1].dtype)
      #print('begin send of', rank )		  
      comm.Recv(recbuf, source=0, tag=100)
      J[:,:,0] = recbuf
      del recbuf	  
      #print('goodsend of', rank )	  
      sendbuf = J[:,:,1].copy()	  
      comm.Send(sendbuf, dest=0, tag=110)
      del sendbuf
      #print('goodsend of', rank )	  
 	  
	   
  if rank ==0:
	# 接受数据
    deimg = np.empty([nx,ny,nz], dtype = np.float64)
    deimg[:,:,0:100] = J[:,:,0:100]
    recbuf = np.empty([200,200,100],np.float64)
    comm.Recv(recbuf, source=1, tag=20)
    deimg[:,:,100:200] = recbuf
    del recbuf  	
    plt.figure()
    plt.imshow(deimg[:,100,:],"gray")
    plt.savefig('./result.png')	
  else:
    sendbuf = J[:,:,1:101].copy()
    comm.Send(sendbuf, dest=0, tag=20)  
    del sendbuf
	
		
	
	
	
	
	
	