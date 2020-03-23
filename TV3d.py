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
	
def isend4coord(sendbuf, destcoord, tag, ncoord):
  if destcoord is None:
    return None
  else:
    destrank = coord2rank(destcoord,ncoord)
    req = comm.isend(sendbuf, dest = destrank, tag=tag)
    return req
  
def irecv4coord(sourcecoord, tag, ncoord):
  if sourcecoord is None:
    return None
  else:
    sourcerank = coord2rank(sourcecoord,ncoord)
    req = comm.irecv(source=sourcerank, tag=tag)
    return req

def Send4coord(data, destcoord, tag, ncoord):
  if destcoord is None:
    return None
  else:
    sendbuf = data.copy()
    destrank = coord2rank(destcoord,ncoord)
    comm.Send(sendbuf, dest=destrank, tag=tag)
    return 1
  
def Recv4coord(data, sourcecoord, tag, ncoord):
  if sourcecoord is None:
    return None
  else:
    recbuf = np.empty(data.shape, dtype=data.dtype)
    sourcerank = coord2rank(sourcecoord,ncoord)
    comm.Recv(recbuf, source=sourcerank, tag=tag)
    data = recbuf
    return 1
	
if __name__ == '__main__':
  from mpi4py import MPI
  T, dt, lam = 50, 0.1, 0.01 
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  # np = 8, ncoord = [2,2,2]
  ncoord = [2,2,2]
  dim = len(ncoord)
  coord = rank2coord(rank, ncoord)
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
    nimgsile = nimg[0:101,0:101,0:101]
    for i in range(1,size):
      icoord = rank2coord(i,ncoord)
      start = dim*[0]
      end = dim*[0]
      for j in range(dim):
        if icoord[j]==0:  
          start[j] = 0
          end[j] = 101
        else:
          start[j] = 99
          end[j] = 200
      sendbuf = nimg[start[0]:end[0], start[1]:end[1], start[2]:end[2]].copy()
      comm.Send(sendbuf, dest=i, tag=11)
    del nimg, sendbuf
  else: 
    nimgsile = np.empty([101,101,101],dtype=np.float64) 
    comm.Recv(nimgsile, source=0, tag=11)
	  

  # Compute the rank for up, down, left, right,  front, behind
  if coord[0]>0:
    upcoord = coord.copy()
    upcoord[0] = coord[0]-1
  else:
    upcoord = None
	
  if coord[0]<ncoord[0]-1:
    downcoord = coord.copy()
    downcoord[0] = coord[0]+1 
  else:
    downcoord = None
  
  if coord[1]>0:
    leftcoord = coord.copy()
    leftcoord[1] = coord[1]-1 
  else:
    leftcoord = None
	
  if coord[1]<ncoord[1]-1:
    rightcoord = coord.copy()
    rightcoord[1] = coord[1]+1
  else:
    rightcoord = None  
  
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
	
 # print(upcoord,downcoord,leftcoord,rightcoord,frontcoord,backcoord, 'of', coord)
  
  # Iterative
  J = nimgsile.copy()
  n0,n1,n2 = J.shape  
  for t in range(T):
    if rank ==0 and not t%5:
      print(t, 'of ', T)
    J = worker(nimgsile, J, dt, lam)
	# Send
    req_send = Send4coord(J[1,:,:], upcoord, 1, ncoord) 
    req_down = Recv4coord(J[n0-1,:,:],downcoord,1, ncoord)	
	
    req_send = Send4coord(J[n0-2,:,:], downcoord, 1, ncoord)
    req_up = Recv4coord(J[0,:,:],upcoord,1, ncoord) 	
	
    req_send = Send4coord(J[:,1,:], leftcoord, 1, ncoord)
    req_right = Recv4coord(J[:,n1-1,:],rightcoord,1, ncoord)	

	
    req_send = Send4coord(J[:,n1-2,:], rightcoord, 1, ncoord)
    req_left = Recv4coord(J[:,0,:],leftcoord,1, ncoord)	
	
    req_send = Send4coord(J[:,:,1], frontcoord, 1, ncoord)
    req_back = Recv4coord(J[:,:,n2-1],backcoord,1, ncoord)

    req_send = Send4coord(J[:,:,n2-2], backcoord, 1, ncoord)
    req_front = Recv4coord(J[:,:,0], frontcoord,1, ncoord)
   
  if rank ==0:
    import matplotlib.pyplot as plt # plt 用于显示图片
	# 接受数据
    deimg = np.empty([nx,ny,nz], dtype = np.float64)
	deimg[0:100,0:100,0:100] = J
    for i in range(1,size):
      icoord = rank2coord(i,ncoord)
      start = dim*[0]
      end = dim*[0]
      for j in range(dim):
        if icoord[j]==0:  
          start[j] = 0
          end[j] = 100
        else:
          start[j] = 100
          end[j] = 200
      recbuf = np.empty([100,100,100],np.float64)
      comm.Recv(recbuf, source=i, tag=20)
      deimg[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = recbuf
    del recbuf  
    plt.title(r'Denoisies Image of TV3D$')
    plt.imshow(deimg[:,:,100],"gray")
    plt.axis('off')
    plt.show()
  else:
    start = dim*[0]
    end = dim*[0]
    for j in range(dim):
      if coord[j] == 0:
        start[j] = 0
        end[j] = 100
      else:
        start[j] = 1
        end[j] = 101
    sendbuf = J[start[0]:end[0],start[1]:end[1],start[2]:end[2]].copy()
    comm.Send(sendbuf, dest=0, tag=20)  
    del sendbuf
	
	
	
	
	
	
	
	
	
	
	
	
	