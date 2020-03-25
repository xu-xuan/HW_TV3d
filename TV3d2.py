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
  DfJx=J[list(range(1,m))+[m-1],:,:]-J
  DfJy=J[:,list(range(1,n))+[n-1],:]-J
  DfJz=J[:,:,list(range(1,l))+[l-1]]-J

  TempDJ=(ep+DfJx*DfJx+DfJy*DfJy+DfJz*DfJz)**(1/2)
  DivJx=DfJx/TempDJ;
  DivJy=DfJy/TempDJ;
  DivJz=DfJz/TempDJ;
	
  Div=bdx(DivJx,m)+bdy(DivJy,n)+bdz(DivJz,l)
          
  J += dt * Div -dt*lam*(J-In)
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
  import matplotlib.pyplot as plt # plt 用于显示图片
  T, dt, lam = 3, 0.1, 0.01 
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
    plt.figure()
    plt.imshow(img[:,100,:],"gray")
    plt.savefig('./img.png')
    plt.figure()
    plt.imshow(nimg[:,100,:],"gray")
    plt.savefig('./nimg.png')	
    del img
    nimgsile = nimg[:,:,0:101]
    sendbuf = nimg[:,:,99:200].copy()
    plt.figure()
    plt.imshow(sendbuf[:,100,:],"gray")
    plt.savefig('./sendbuf.png')	
    comm.Send(sendbuf, dest=1, tag=11)
    plt.figure()
    plt.imshow(nimgsile[:,100,:],"gray")
    plt.savefig('./nimgsile0.png')	
    del nimg, sendbuf
  else: 
    nimgsile = np.empty([200,200,101],dtype=np.float64) 
    comm.Recv(nimgsile, source=0, tag=11)
    plt.figure()
    plt.imshow(nimgsile[:,100,:],"gray")
    plt.savefig('./nimgsile1.png')	

  # if coord[2]>0:
    # frontcoord = coord.copy()
    # frontcoord[2] = coord[2]-1
  # else: 
    # frontcoord = None
	
  # if coord[2]<ncoord[2]-1:
    # backcoord = coord.copy()
    # backcoord[2] = coord[2]+1
  # else:
    # backcoord = None 
  # #print(frontcoord,backcoord, 'of', coord, rank)
  # Iterative
  J = nimgsile.copy()
  n0,n1,n2 = J.shape  
  print(n2, 'of', rank )
  for t in range(T):
    if rank ==0 and not t%5:
      print(t, 'of ', T)
    J = worker(nimgsile, J, dt, lam)
	if rank == 0:
      sendbuf = J[:,:,n2-1].copy()
      comm.Send(sendbuf, dest=1, tag=100)
      recbuf = np.empty(data.shape, dtype=data.dtype)
      sourcerank = coord2rank(sourcecoord,ncoord)
      comm.Recv(recbuf, source=sourcerank, tag=tag)      

    Send4coord(J[:,:,n2-2], backcoord, 1, ncoord)
    J[:,:,0] = Recv4coord(J[:,:,0], frontcoord,1, ncoord)		
	
   
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
	
	
	
	
	
	
	
	
	
	
	
	
	