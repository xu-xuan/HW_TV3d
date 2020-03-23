# echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
import numpy as np

def bdx(P,m):
  return P-P[[0]+list(range(m-1)),:,:]

def bdy(P,n):
  return P-P[:,[0]+list(range(n-1)),:]    

def bdz(P,l):
  return P-P[:,:,[0]+list(range(l-1))]	
	
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

def rank2coord(rank, ncoord):
  # warning： ncoord == [2,2,2]
  c = '{:03b}'.format(int(rank))
  coord = [eval(c[i]) for i in range(3)]
  return coord

def coord2rank(coord, ncoord):
  # warning： ncoord[i] == 2
  dim = len(ncoord)
  rank = 0
  for i in range(dim):
    rank += coord[i]*(2**(dim-i-1))
  return rank
	

def Send4coord(data, destcoord, tag, ncoord):
  if destcoord is None:
    return None
  else:
    sendbuf = data.copy()
    destrank = coord2rank(destcoord,ncoord)
    #print(destrank)
    comm.Send(sendbuf, dest=destrank, tag=tag)
    del sendbuf
    return 1
  
def Recv4coord(data, sourcecoord, tag, ncoord):
  if sourcecoord is None:
    return None
  else:
    recbuf = np.empty(data.shape, dtype=data.dtype)
    sourcerank = coord2rank(sourcecoord,ncoord)
    #print(sourcerank)
    comm.Recv(recbuf, source=sourcerank, tag=tag)
    return recbuf	
	
if __name__ == '__main__':
  from mpi4py import MPI
  T, dt, lam = 50, 0.1, 0.01 
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  ncoord = [1,1,2]
  coord = rank2coord(rank, ncoord)
  
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
    del img
    nimgsile = nimg[:,:,0:101]
    sendbuf = nimg[:,:,100:200].copy()
    comm.Send(sendbuf, dest=1, tag=11)
    del nimg, sendbuf
  else: 
    nimgsile = np.empty([200,200,101],dtype=np.float64) 
    comm.Recv(nimgsile, source=0, tag=11)

  if coord[2]>0:
    frontcoord = coord.copy()
    frontcoord[2] = coord[2]-1
  else: 
    frontcoord = None
	
  if coord[2]<ncoord[2]-1:
    backcoord = coord.copy()
    backcoord[2] = coord[2]+1
  else:
    backcoord = None 
  #print(frontcoord,backcoord, 'of', coord, rank)
  # Iterative
  J = nimgsile.copy()
  n0,n1,n2 = J.shape  
  for t in range(T):
    if rank ==0 and not t%5:
      print(t, 'of ', T)
    J = worker(nimgsile, J, dt, lam)
	# Send
    Send4coord(J[:,:,1], frontcoord, 1, ncoord)
    J[:,:,n2-1] =  Recv4coord(J[:,:,n2-1],backcoord,1, ncoord)

    Send4coord(J[:,:,n2-2], backcoord, 1, ncoord)
    J[:,:,0] = Recv4coord(J[:,:,0], frontcoord,1, ncoord)		
	
   
  if rank ==0:
    import matplotlib.pyplot as plt # plt 用于显示图片
	# 接受数据
    deimg = np.empty([nx,ny,nz], dtype = np.float64)
    deimg[:,:,0:100] = J[:,:,0:100]
    recbuf = np.empty([200,200,100],np.float64)
    comm.Recv(recbuf, source=1, tag=20)
    deimg[:,:,100:200] = recbuf
    del recbuf  
    plt.title(r'Denoisies Image of TV3D$')
    plt.imshow(deimg[:,100,:],"gray")
    plt.axis('off')
    plt.show()
  else:
    sendbuf = J[:,:,1:101].copy()
    comm.Send(sendbuf, dest=0, tag=20)  
    del sendbuf
	
	
	
	
	
	
	
	
	
	
	
	
	