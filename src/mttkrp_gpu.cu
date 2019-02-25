/**
 *
 * OHIO STATE UNIVERSITY SOFTWARE DISTRIBUTION LICENSE
 *
 * Load-balanced sparse MTTKRP on GPUs (the “Software”) Copyright (c) 2019, The Ohio State
 * University. All rights reserved.
 *
 * The Software is available for download and use subject to the terms and
 * conditions of this License. Access or use of the Software constitutes acceptance
 * and agreement to the terms and conditions of this License. Redistribution and
 * use of the Software in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the capitalized paragraph below.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the capitalized paragraph below in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. The names of Ohio State University, or its faculty, staff or students may not
 * be used to endorse or promote products derived from the Software without
 * specific prior written permission.
 *
 * THIS SOFTWARE HAS BEEN APPROVED FOR PUBLIC RELEASE, UNLIMITED DISTRIBUTION. THE
 * SOFTWARE IS PROVIDED “AS IS” AND WITHOUT ANY EXPRESS, IMPLIED OR STATUTORY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, WARRANTIES OF ACCURACY, COMPLETENESS,
 * NONINFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  ACCESS OR USE OF THE SOFTWARE IS ENTIRELY AT THE USER’S RISK.  IN
 * NO EVENT SHALL OHIO STATE UNIVERSITY OR ITS FACULTY, STAFF OR STUDENTS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  THE SOFTWARE
 * USER SHALL INDEMNIFY, DEFEND AND HOLD HARMLESS OHIO STATE UNIVERSITY AND ITS
 * FACULTY, STAFF AND STUDENTS FROM ANY AND ALL CLAIMS, ACTIONS, DAMAGES, LOSSES,
 * LIABILITIES, COSTS AND EXPENSES, INCLUDING ATTORNEYS’ FEES AND COURT COSTS,
 * DIRECTLY OR INDIRECTLY ARISING OUT OF OR IN CONNECTION WITH ACCESS OR USE OF THE
 * SOFTWARE.
 *
 */

/**
 *
 * Author:
 *          Israt Nisa (nisa.1@osu.edu)
 *
 * Contacts:
 *          Israt Nisa (nisa.1@osu.edu)
 *          Jiajia Li (jiajia.li@pnnl.gov)
 *          Aravind Sukumaran-Rajam (sukumaranrajam.1@osu.edu)
 *          P.(Saday) Sadayappan (sadayappan.1@osu.edu)
 *
 */

#include <iostream>
#include "mttkrp_gpu.h"
#include <vector>

inline cudaError_t checkCuda(cudaError_t result, int s){

  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error in line : %s - %d\n", cudaGetErrorString(result), s);
    assert(result == cudaSuccess);
  }
  return result;
}

void cuda_timer_start(cudaEvent_t start){
	checkCuda(cudaEventRecord(start), __LINE__);
}
void cuda_timer_stop(cudaEvent_t start, cudaEvent_t stop, float &mili){
	checkCuda(cudaEventRecord(stop), __LINE__);
    cudaEventSynchronize(stop);
    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
    cudaDeviceSynchronize();
}

// CUDA kernel call to do COO MTTKRP 
__global__ void mttkrp_COO_kernel(DTYPE *vals, ITYPE *dInds0, ITYPE *dInds1, ITYPE *dInds2,  ITYPE nnz,
	DTYPE *dU0, DTYPE *dU1, DTYPE *dU2, ITYPE	mode, ITYPE R){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int x = gId >> 5;
	
	if(x < nnz){
        DTYPE tmp_val = 0;
        ITYPE idx0 = dInds0[x];
        ITYPE idx1 = dInds1[x];
        ITYPE idx2 = dInds2[x];

        for(ITYPE r=laneId; r<R; r+=32) {           
            tmp_val = vals[x] * dU1[idx1 * R + r] * dU2[idx2 * R + r];
            atomicAdd(&dU0[idx0 * R + r], tmp_val);
        }    
	}
}

// CUDA kernel call to do COO MTTKRP using loop
__global__ void mttkrp_COO_kernel_loop(DTYPE * const vals, ITYPE * const dInds0, ITYPE * const dInds1, ITYPE * const dInds2,  const ITYPE nnz,
	DTYPE *dU0, DTYPE * const dU1, DTYPE * const dU2, ITYPE	mode, ITYPE R){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);

	//like PARTI
	size_t num_loops_nnz = 1 * 32;
    size_t const nnz_per_loop = gridDim.x * blockDim.x;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = ((nnz + nnz_per_loop - 1) / nnz_per_loop) << 5;
    }

	unsigned int x;

	for(size_t nl=0; nl<num_loops_nnz; ++nl) {
		
		x = (gId + nl * nnz_per_loop) >> 5;
		
		if(x < nnz){
	    
	        DTYPE tmp_val = 0;
	        ITYPE idx0 = dInds0[x];
	        ITYPE idx1 = dInds1[x];
	        ITYPE idx2 = dInds2[x];

	        for(ITYPE r=laneId; r<R; r+=32) {           
	            tmp_val = vals[x] * dU1[idx1 * R + r] * dU2[idx2 * R + r];
	            atomicAdd(&dU0[idx0 * R + r], tmp_val);
	        }  
		}
		__syncthreads();
	}
}
// CUDA kernel call to do COO MTTKRP 4D 
__global__ void mttkrp_COO_kernel_4D(DTYPE *vals, ITYPE *dInds0, ITYPE *dInds1, ITYPE *dInds2, ITYPE *dInds3,
    ITYPE nnz, DTYPE *dU0, DTYPE *dU1, DTYPE *dU2,  DTYPE *dU3, ITYPE mode, ITYPE R){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int x = gId >> 5;
	
	if(x < nnz){
        DTYPE tmp_val = 0;
        ITYPE idx0 = dInds0[x];
        ITYPE idx1 = dInds1[x];
        ITYPE idx2 = dInds2[x];
        ITYPE idx3 = dInds3[x];

        for(ITYPE r=laneId; r<R; r+=32) {           
            tmp_val = vals[x] * dU1[idx1 * R + r] * dU2[idx2 * R + r]  * dU3[idx3 * R + r];
            atomicAdd(&dU0[idx0 * R + r], tmp_val);
        }    
	}
}

// CUDA kernel call to do COO MTTKRP 4D using loop
__global__ void mttkrp_COO_kernel_4D_loop(DTYPE *const vals, ITYPE * const dInds0, ITYPE * const dInds1, ITYPE *const dInds2, ITYPE * const dInds3,
    ITYPE nnz, DTYPE *dU0, DTYPE * const dU1, DTYPE * const dU2,  DTYPE * const dU3, ITYPE mode, ITYPE R){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	
	//like PARTI
	size_t num_loops_nnz = 1 * 32;
    size_t const nnz_per_loop = gridDim.x * blockDim.x;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = ((nnz + nnz_per_loop - 1) / nnz_per_loop) << 5;
    }
	unsigned int x;

	for(size_t nl=0; nl<num_loops_nnz; ++nl) 
	{
		x = (gId + nl * nnz_per_loop) >> 5;

		if(x < nnz){
	        DTYPE tmp_val = 0;
	        ITYPE idx0 = dInds0[x];
	        ITYPE idx1 = dInds1[x];
	        ITYPE idx2 = dInds2[x];
	        ITYPE idx3 = dInds3[x];

	        for(ITYPE r=laneId; r<R; r+=32) {           
	            tmp_val = vals[x] * dU1[idx1 * R + r] * dU2[idx2 * R + r]  * dU3[idx3 * R + r];
	            atomicAdd(&dU0[idx0 * R + r], tmp_val);
	        }
	    }  
	    __syncthreads();  
	}
}
//no atomics because all 1 in HYB - COO 
__global__ void mttkrp_HYB_COO_kernel(DTYPE *vals, ITYPE *dInds0, ITYPE *dInds1, ITYPE *dInds2,  ITYPE nnz,
	DTYPE *dU0, DTYPE *dU1, DTYPE *dU2, ITYPE	mode, ITYPE R){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int x = gId >> 5;
	
	if(x < nnz){
        DTYPE tmp_val = 0;
        ITYPE idx0 = dInds0[x];
        ITYPE idx1 = dInds1[x];
        ITYPE idx2 = dInds2[x];

        for(ITYPE r=laneId; r<R; r+=32) {           
            tmp_val = vals[x] * dU1[idx1 * R + r] * dU2[idx2 * R + r];
            dU0[idx0 * R + r] += tmp_val;
        }    
	}
}

// CUDA kernel call to do COO MTTKRP using loop
__global__ void mttkrp_HYB_COO_kernel_loop(DTYPE * const vals, ITYPE * const dInds0, ITYPE * const dInds1, ITYPE * const dInds2,  const ITYPE nnz,
	DTYPE *dU0, DTYPE * const dU1, DTYPE * const dU2, ITYPE	mode, ITYPE R){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);

	//like PARTI
	size_t num_loops_nnz = 1 * 32;
    size_t const nnz_per_loop = gridDim.x * blockDim.x;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = ((nnz + nnz_per_loop - 1) / nnz_per_loop) << 5;
    }

	unsigned int x;

	for(size_t nl=0; nl<num_loops_nnz; ++nl) {
		
		x = (gId + nl * nnz_per_loop) >> 5;
		
		if(x < nnz){
	    
	        DTYPE tmp_val = 0;
	        ITYPE idx0 = dInds0[x];
	        ITYPE idx1 = dInds1[x];
	        ITYPE idx2 = dInds2[x];

	        for(ITYPE r=laneId; r<R; r+=32) {           
	            tmp_val = vals[x] * dU1[idx1 * R + r] * dU2[idx2 * R + r];
	            dU0[idx0 * R + r] += tmp_val;
	        }  
		}
		__syncthreads();
	}
}

//no atomics because all 1 in HYB - COO 
__global__ void mttkrp_HYB_COO_kernel_4D(DTYPE *vals, ITYPE *dInds0, ITYPE *dInds1, ITYPE *dInds2, ITYPE *dInds3,
  ITYPE nnz,  DTYPE *dU0, DTYPE *dU1, DTYPE *dU2,  DTYPE *dU3, ITYPE mode, ITYPE R){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int x = gId >> 5;
	
	if(x < nnz){
        DTYPE tmp_val = 0;
        ITYPE idx0 = dInds0[x];
        ITYPE idx1 = dInds1[x];
        ITYPE idx2 = dInds2[x];
        ITYPE idx3 = dInds3[x];

        for(ITYPE r=laneId; r<R; r+=32) {           
            tmp_val = vals[x] * dU1[idx1 * R + r] * dU2[idx2 * R + r] * dU3[idx3 * R + r];
            dU0[idx0 * R + r] += tmp_val;
        }    
	}
}

// CUDA kernel call to do COO MTTKRP 4D using loop
__global__ void mttkrp_HYB_COO_kernel_4D_loop(DTYPE *const vals, ITYPE * const dInds0, ITYPE * const dInds1, ITYPE *const dInds2, ITYPE * const dInds3,
    ITYPE nnz, DTYPE *dU0, DTYPE * const dU1, DTYPE * const dU2,  DTYPE * const dU3, ITYPE mode, ITYPE R){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	
	//like PARTI
	size_t num_loops_nnz = 1 * 32;
    size_t const nnz_per_loop = gridDim.x * blockDim.x;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = ((nnz + nnz_per_loop - 1) / nnz_per_loop) << 5;
    }
	unsigned int x;

	for(size_t nl=0; nl<num_loops_nnz; ++nl) 
	{
		x = (gId + nl * nnz_per_loop) >> 5;

		if(x < nnz){
	        DTYPE tmp_val = 0;
	        ITYPE idx0 = dInds0[x];
	        ITYPE idx1 = dInds1[x];
	        ITYPE idx2 = dInds2[x];
	        ITYPE idx3 = dInds3[x];

	        for(ITYPE r=laneId; r<R; r+=32) {           
	            tmp_val = vals[x] * dU1[idx1 * R + r] * dU2[idx2 * R + r]  * dU3[idx3 * R + r];
	            dU0[idx0 * R + r] += tmp_val;
	        }
	    }  
	    __syncthreads();  
	}
}

__global__ void mttkrp_CSL_kernel(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
	ITYPE *dInds1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  
	unsigned int slc = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) 
	DTYPE tmp_val;
		              	              
	if(slc < nSlices){ 	    

		unsigned int mappedSlc = slc;//dSlcMapperBin[slc];
		unsigned int idx0 = dfbrIdx0[mappedSlc]; 
    	int fb_st = fbrPtr0[mappedSlc];
		int fb_end = fbrPtr0[mappedSlc+1];
		tmp_val = 0;
		
		for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
			
		    unsigned int idx1 = dInds1[fbr];
	        unsigned int idx2 = dInds2[fbr];                
            for(unsigned int r=laneId; r<R; r+=32) {
                tmp_val += vals[fbr] * dU2[idx2 * R + r] * dU1[idx1 * R + r]; 
            }   
		}
		for(unsigned int r=laneId; r<R; r+=32) {  
			atomicAdd(&dU0[idx0 * R + r], tmp_val);    
		}
	}
}

__global__ void mttkrp_CSL_kernel_bin(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
	ITYPE *dInds1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  
	unsigned int slc = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) 
	DTYPE tmp_val;
		              	              
	if(slc < nSlices){ 	    

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx0 = dfbrIdx0[mappedSlc]; 
    	int fb_st = fbrPtr0[mappedSlc];
		int fb_end = fbrPtr0[mappedSlc+1];
		tmp_val = 0;
		
		for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
			
		    unsigned int idx1 = dInds1[fbr];
	        unsigned int idx2 = dInds2[fbr];                
            for(unsigned int r=laneId; r<R; r+=32) {
                tmp_val += vals[fbr] * dU2[idx2 * R + r] * dU1[idx1 * R + r]; 
            }   
		}
		for(unsigned int r=laneId; r<R; r+=32) {  
			atomicAdd(&dU0[idx0 * R + r], tmp_val);    
		}
	}
}

// CSL kernel with loop like ParTI
__global__ void mttkrp_CSL_kernel_bin_loop(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
	ITYPE *dInds1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  
	unsigned int slc = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) 
	DTYPE tmp_val;

	//like PARTI
	size_t num_loops_nnz = 1 * 32;
    size_t const nnz_per_loop = gridDim.x * blockDim.x;
    if(nSlices > nnz_per_loop) {
        num_loops_nnz = ((nSlices + nnz_per_loop - 1) / nnz_per_loop) << 5;
    }

	for(size_t nl=0; nl<num_loops_nnz; ++nl) {
		
		slc = (gId + nl * nnz_per_loop) >> 5;
		              	              
		if(slc < nSlices){ 	    

			unsigned int mappedSlc = dSlcMapperBin[slc];
			unsigned int idx0 = dfbrIdx0[mappedSlc]; 
	    	int fb_st = fbrPtr0[mappedSlc];
			int fb_end = fbrPtr0[mappedSlc+1];
			tmp_val = 0;
			
			for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
				
			    unsigned int idx1 = dInds1[fbr];
		        unsigned int idx2 = dInds2[fbr];                
	            for(unsigned int r=laneId; r<R; r+=32) {
	                tmp_val += vals[fbr] * dU2[idx2 * R + r] * dU1[idx1 * R + r]; 
	            }   
			}
			for(unsigned int r=laneId; r<R; r+=32) {  
				atomicAdd(&dU0[idx0 * R + r], tmp_val);    
			}
		}
		__syncthreads();  
	}
}

// CUDA kernel call to do HCSR MTTKRP 
__global__ void mttkrp_CSL_kernel_hvyBin(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
	ITYPE *dInds1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int logOfTPS){
	
	unsigned int laneId = threadIdx.x & 31;
	unsigned int workId = threadIdx.x >> 5;
	unsigned int slc = blockIdx.x >> logOfTPS;
	unsigned int localBId = blockIdx.x & (TbPerSlc -1);
	
	DTYPE tmp = 0, tmp_val;
		              	              
	if(slc < nSlices){

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx0 = dfbrIdx0[mappedSlc] ;//slc;
		unsigned int nFbr = fbrPtr0[mappedSlc+1] - fbrPtr0[mappedSlc];		
		unsigned int fbrPerTb = (nFbr + TbPerSlc - 1 ) >> logOfTPS; 
		unsigned int fb_st = fbrPtr0[mappedSlc] + localBId * fbrPerTb ;
		unsigned int fb_end = fbrPtr0[mappedSlc] + (localBId + 1) * fbrPerTb ;

		tmp_val = 0;
		for (int fbr = fb_st + workId; fbr < fb_end && fbr < fbrPtr0[mappedSlc+1]; fbr+=warpPerSlice){
			
		    unsigned int idx1 = dInds1[fbr];
	        unsigned int idx2 = dInds2[fbr];                
            for(unsigned int r=laneId; r<R; r+=32) {
                tmp_val += vals[fbr] * dU2[idx2 * R + r] * dU1[idx1 * R + r]; 
            }   
		}
		for(unsigned int r=laneId; r<R; r+=32) {  
			atomicAdd(&dU0[idx0 * R + r], tmp_val);    
		} 
	}
}

// HCSR MTTKRP : 16 WARP = 1 TB per slice
__global__ void mttkrp_HCSR_kernel_16WARP(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
	ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = tId >> 5; //(tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  
	unsigned int slc = blockIdx.x ;//gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) 
	DTYPE tmp = 0; 
	DTYPE tmp_val;
		              	              
	if(slc < nSlices){ 	    

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx0 = dfbrIdx0[mappedSlc] ;//slc;
    	int fb_st = fbrPtr0[mappedSlc];
		int fb_end = fbrPtr0[mappedSlc+1];

		for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
			tmp_val = 0;
	        for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

		        unsigned int idx2 = dInds2[x];                
	            for(unsigned int r=laneId; r<R; r+=32) {
	                tmp_val += vals[x] * dU2[idx2 * R + r]; 
	            }
	        }
	        // unsigned int idx1 = dInds1[fbrPtr1[fbr]]; 
	        unsigned int idx1 = fbrIdx1[fbr];   
	        for(unsigned int r=laneId; r<R; r+=32) {  
	        	tmp += tmp_val * dU1[idx1 * R + r] ;     
	        }    
		}

		for(unsigned int r=laneId; r<R; r+=32) {  
            atomicAdd(&dU0[idx0 * R + r], tmp);
        } 
	}
}
// CUDA kernel call to do HCSR MTTKRP for the first bin 1 WARP per slice
__global__ void mttkrp_HCSR_kernel_COO(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
	ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int slc = gId >> 5; // 5: minimum 1 WARP (2^5) 
	DTYPE tmp = 0, tmp_val;
		              	              
	if(slc < nSlices){ 	    

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx0 = dfbrIdx0[mappedSlc] ;//slc;
    	int fb_st = fbrPtr0[mappedSlc];
		int fb_end = fbrPtr0[mappedSlc+1];

		for (int fbr = fb_st; fbr < fb_end; fbr++){
			tmp_val = 0;
	        for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

		        unsigned int idx2 = dInds2[x];                
	            for(unsigned int r=laneId; r<R; r+=32) {
	                tmp_val += vals[x] * dU2[idx2 * R + r]; 
	            }
	        }
	        unsigned int idx1 = fbrIdx1[fbr];
	        for(unsigned int r=laneId; r<R; r+=32) {  
	        	dU0[idx0 * R + r] += tmp_val * dU1[idx1 * R + r] ;     
	        }    
		}
	}
}

// CUDA kernel call to do HCSR MTTKRP 
__global__ void mttkrp_HCSR_kernel_smllBin(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
	ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  
	unsigned int slc = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) 
	// unsigned int slcPerTb = 16/warpPerSlice;
	// unsigned int shSlc = slc & slcPerTb;
	DTYPE tmp = 0, tmp_val;
		              	              
	if(slc < nSlices){ 	    

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx0 = dfbrIdx0[mappedSlc] ;//slc;
    	int fb_st = fbrPtr0[mappedSlc];
		int fb_end = fbrPtr0[mappedSlc+1];

		for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
			tmp_val = 0;
	        for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

		        unsigned int idx2 = dInds2[x];                
	            for(unsigned int r=laneId; r<R; r+=32) {
	                tmp_val += vals[x] * dU2[idx2 * R + r]; 
	            }
	        }
	        unsigned int idx1 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];    
	        for(unsigned int r=laneId; r<R; r+=32) {  
	        	tmp += tmp_val * dU1[idx1 * R + r] ;     
	        }    
		}

		for(unsigned int r=laneId; r<R; r+=32) {  
			atomicAdd(&dU0[idx0 * R + r], tmp);       
		}
	}
}

// CUDA kernel call to do HCSR MTTKRP 
__global__ void mttkrp_HCSR_kernel_smllBin_4D(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds3, ITYPE *fbrPtr0,
	ITYPE *fbrPtr1, ITYPE *fbrIdx1, ITYPE *fbrPtr2, ITYPE *fbrIdx2, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, DTYPE *dU3, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  
	unsigned int slc = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) 
	DTYPE outbuffer = 0, tmp_val = 0, outbuffer1 = 0;
		              	              
	if(slc < nSlices){ 	    

		unsigned int mappedSlc = dSlcMapperBin[slc];
		
		unsigned int idx0 = dfbrIdx0[mappedSlc] ;//slc;

		for (int fbrS = fbrPtr0[mappedSlc]; fbrS < fbrPtr0[mappedSlc+1]; fbrS++){
			
			unsigned int idx1 = fbrIdx1[fbrS];// dInds1[fbrPtr1[fbr]];  
			outbuffer1 = 0;
			
			for (int fbr = fbrPtr1[fbrS] + workId; fbr < fbrPtr1[fbrS+1]; fbr+=warpPerSlice){
				ITYPE idx2 = fbrIdx2[fbr];
				tmp_val = 0;
	    
		        for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; ++x) {

			        unsigned int idx3 = dInds3[x];                
		            for(unsigned int r=laneId; r<R; r+=32) 
		                tmp_val += vals[x] * dU3[idx3 * R + r]; 
		        }       
		        for(unsigned int r=laneId; r<R; r+=32)  
		        	outbuffer1 += tmp_val * dU2[idx2 * R + r] ;       
		    }
		    for(unsigned int r=laneId; r<R; r+=32) 
	        	outbuffer += outbuffer1 * dU1[idx1 * R + r] ;    
		}
		for(unsigned int r=laneId; r<R; r+=32) {  
			atomicAdd(&dU0[idx0 * R + r], outbuffer);  
		}
	}
}

// CUDA kernel call to do HCSR MTTKRP 
__global__ void mttkrp_HCSR_kernel_hvyBin(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
	ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int logOfTPS){
	
	unsigned int laneId = threadIdx.x & 31;
	unsigned int workId = threadIdx.x >> 5;
	unsigned int slc = blockIdx.x >> logOfTPS;
	unsigned int localBId = blockIdx.x & (TbPerSlc -1);
	
	DTYPE tmp = 0, tmp_val;
		              	              
	if(slc < nSlices){

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx0 = dfbrIdx0[mappedSlc] ;//slc;
		unsigned int nFbr = fbrPtr0[mappedSlc+1] - fbrPtr0[mappedSlc];		
		unsigned int fbrPerTb = (nFbr + TbPerSlc - 1 ) >> logOfTPS; 
		unsigned int fb_st = fbrPtr0[mappedSlc] + localBId * fbrPerTb ;
		unsigned int fb_end = fbrPtr0[mappedSlc] + (localBId + 1) * fbrPerTb ;

		for (int fbr = fb_st + workId; fbr < fb_end && fbr < fbrPtr0[mappedSlc+1] ; fbr+=warpPerSlice){
			tmp_val = 0;
	        
	        for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

		        unsigned int idx2 = dInds2[x];                
	            for(unsigned int r=laneId; r<R; r+=32) {
	                tmp_val += vals[x] * dU2[idx2 * R + r]; 
	            }
	        }
	        unsigned int idx1 = fbrIdx1[fbr];//dInds1[fbrPtr1[fbr]];    
	        for(unsigned int r=laneId; r<R; r+=32) {  
	        	tmp += tmp_val * dU1[idx1 * R + r] ;     
	            // // atomicAdd(&dU0[idx0 * R + r], tmp);
	        }    
		}
		for(unsigned int r=laneId; r<R; r+=32) {  
            atomicAdd(&dU0[idx0 * R + r], tmp);
        } 
	}
}

// CUDA kernel call to do HCSR MTTKRP 
__global__ void mttkrp_HCSR_kernel_hvyBin_4D(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds3, ITYPE *fbrPtr0,
	ITYPE *fbrPtr1, ITYPE *fbrIdx1, ITYPE *fbrPtr2, ITYPE *fbrIdx2, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, DTYPE *dU3, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int logOfTPS){
	
	unsigned int laneId = threadIdx.x & 31;
	unsigned int workId = threadIdx.x >> 5;
	unsigned int slc = blockIdx.x >> logOfTPS;
	unsigned int localBId = blockIdx.x & (TbPerSlc -1);
	
	DTYPE outbuffer = 0, tmp_val = 0, outbuffer1 = 0;;
		              	              
	if(slc < nSlices){

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx0 = dfbrIdx0[mappedSlc] ;//slc;
		unsigned int nFbr = fbrPtr0[mappedSlc+1] - fbrPtr0[mappedSlc];		
		unsigned int fbrPerTb = (nFbr + TbPerSlc - 1 ) >> logOfTPS; 
		unsigned int fb_st = fbrPtr0[mappedSlc] + localBId * fbrPerTb ;
		unsigned int fb_end = fbrPtr0[mappedSlc] + (localBId + 1) * fbrPerTb ;

		for (int fbrS = fb_st; fbrS < fb_end && fbrS < fbrPtr0[mappedSlc+1] ; fbrS++){
			unsigned int idx1 = fbrIdx1[fbrS];// dInds1[fbrPtr1[fbr]];  
			outbuffer1 = 0;

			for (int fbr = fbrPtr1[fbrS] + workId; fbr < fbrPtr1[fbrS+1]; fbr+=warpPerSlice){
				ITYPE idx2 = fbrIdx2[fbr];
				tmp_val = 0;
            
		        for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; ++x) {

			        unsigned int idx3 = dInds3[x];                
		            for(unsigned int r=laneId; r<R; r+=32) 
		                tmp_val += vals[x] * dU3[idx3 * R + r]; 
		        }
		        for(unsigned int r=laneId; r<R; r+=32)  
		        	outbuffer1 += tmp_val * dU2[idx2 * R + r] ;  
		    }
		    for(unsigned int r=laneId; r<R; r+=32) 
	        	outbuffer += outbuffer1 * dU1[idx1 * R + r] ;     
		}
		for(unsigned int r=laneId; r<R; r+=32) { 
            atomicAdd(&dU0[idx0 * R + r], outbuffer);
        } 
	}
}

// CUDA fbr atomic sing slcLikeFbr
__global__ void mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar(DTYPE * vals, ITYPE *fbrLikeSlcInds, ITYPE *dInds2, 
	ITYPE *fbrPtr0, ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nFibers, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
	unsigned int fbr = (gId >> (5 + logOfWPC)) << 2; // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
	DTYPE tmp = 0, tmp_val;
		              	              
	if(fbr < nFibers - 1){ 	    
		
		tmp_val = 0;
		bool diffFiber = false;
		unsigned int idx0;

		//maybe add another check fbr_fr can be larger than fiber?

		for (int fr = 0; fr < 4 && (fbr+fr) < (nFibers - 1); ++fr){

			diffFiber = false;
			unsigned int idx1 = fbrIdx1[fbr+fr];// dInds1[fbrPtr1[fbr]];  
			idx0 = fbrLikeSlcInds[fbr+fr];//slc;  
 			tmp_val = 0;
	        for(unsigned int x = fbrPtr1[fbr+fr] + workId; x < fbrPtr1[fbr+fr+1]; x+=warpPerSlice) {

		        unsigned int idx2 = dInds2[x];                    

	            for(unsigned int r=laneId; r<R; r+=32) {
	                tmp_val += vals[x] * dU2[idx2 * R + r]; //2MR   
	            }       
	        }
	        	
        	for(unsigned int r=laneId; r<R; r+=32) { 
        		tmp += tmp_val * dU1[idx1 * R + r] ;
        	} 
	        
        	if(fbrLikeSlcInds[fbr+fr] != fbrLikeSlcInds[fbr+fr+1]) {

        		diffFiber = true;
	        	for(unsigned int r=laneId; r<R; r+=32) { 
	        		atomicAdd(&dU0[idx0 * R + r], tmp); //2PR
	        	} 
        		tmp = 0;
        	}
        } 

        if(!diffFiber) {  
	        for(unsigned int r=laneId; r<R; r+=32) { 
	        	atomicAdd(&dU0[idx0 * R + r], tmp); //2PR
	        }  
        }  
	}
}

// CUDA kernel call to do HCSR MTTKRP 
__global__ void mttkrp_MIHCSR_kernel_smllBin_fbr_atomic(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
	ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  
	unsigned int slc = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) 
	DTYPE tmp = 0, tmp_val;
		              	              
	if(slc < nSlices){ 	    

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx2 = dfbrIdx0[mappedSlc] ;//slc;
    	int fb_st = fbrPtr0[mappedSlc];
		int fb_end = fbrPtr0[mappedSlc+1];

		for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
			
			tmp_val = 0;
			unsigned int idx0 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];    
	        
	        for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

		        unsigned int idx1 = dInds2[x];                    

	            for(unsigned int r=laneId; r<R; r+=32) {
	                tmp_val += vals[x] * dU1[idx1 * R + r]; //2MR
	            }
	        }     
	        for(unsigned int r=laneId; r<R; r+=32) { 
	        	tmp = tmp_val * dU2[idx2 * R + r] ;
	        	atomicAdd(&dU0[idx0 * R + r], tmp); //2PR
	        }    
		}
	}
}

// CUDA fbr atomic sing slcLikeFbr
__global__ void mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar(DTYPE * vals, ITYPE *fbrLikeSlcInds, ITYPE *dInds2, 
	ITYPE *fbrPtr0, ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nFibers, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
	unsigned int fbr = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
	DTYPE tmp = 0, tmp_val;
		              	              
	if(fbr < nFibers - 1){ 	    
		
		tmp_val = 0;
		unsigned int idx0 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];  
		unsigned int idx2 = fbrLikeSlcInds[fbr];//slc;  
        
        for(unsigned int x = fbrPtr1[fbr] + workId; x < fbrPtr1[fbr+1]; x+=warpPerSlice) {

	        unsigned int idx1 = dInds2[x];                    

            for(unsigned int r=laneId; r<R; r+=32) {
                tmp_val += vals[x] * dU1[idx1 * R + r]; //2MR
            }
        }     
        for(unsigned int r=laneId; r<R; r+=32) { 
        	tmp = tmp_val * dU2[idx2 * R + r] ;
        	atomicAdd(&dU0[idx0 * R + r], tmp); //2PR
        }    
	}
}

// CUDA fbr atomic sing slcLikeFbr
__global__ void mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar_loop(DTYPE * vals, ITYPE *fbrLikeSlcInds, ITYPE *dInds2, 
	ITYPE *fbrPtr0, ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nFibers, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);

	//like PARTI
	//hardcoded for 1 warp per nnz
	size_t num_loops_fbr = 1 * 32;
    size_t const fbr_per_loop = gridDim.x * blockDim.x;
    if(nFibers > fbr_per_loop) {
        num_loops_fbr = ((nFibers + fbr_per_loop - 1) / fbr_per_loop) << 5;
    }

	DTYPE tmp = 0, tmp_val;

	unsigned int fbr;

	for(size_t nl=0; nl<num_loops_fbr; ++nl) {
		
		fbr = (gId + nl * fbr_per_loop) >> 5;
		              	              
		if(fbr < nFibers - 1){ 	    
			
			tmp_val = 0;
			unsigned int idx0 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];  
			unsigned int idx2 = fbrLikeSlcInds[fbr];//slc;  
	        
	        for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; x++) {

		        unsigned int idx1 = dInds2[x];                    

	            for(unsigned int r=laneId; r<R; r+=32) {
	                tmp_val += vals[x] * dU1[idx1 * R + r]; //2MR
	            }
	        }     
	        for(unsigned int r=laneId; r<R; r+=32) { 
	        	tmp = tmp_val * dU2[idx2 * R + r] ;
	        	atomicAdd(&dU0[idx0 * R + r], tmp); //2PR
	        }    
		}
	}
}

// CUDA kernel call to do HCSR MTTKRP 
__global__ void mttkrp_MIHCSR_kernel_hvyBin_fbr_atomic(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
	ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int logOfTPS){
	
	unsigned int laneId = threadIdx.x & 31;
	unsigned int workId = threadIdx.x >> 5;
	unsigned int slc = blockIdx.x >> logOfTPS;
	unsigned int localBId = blockIdx.x & (TbPerSlc -1);
	
	DTYPE tmp = 0, tmp_val;
		              	              
	if(slc < nSlices){

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx2 = dfbrIdx0[mappedSlc] ;//slc;
		unsigned int nFbr = fbrPtr0[mappedSlc+1] - fbrPtr0[mappedSlc];		
		unsigned int fbrPerTb = (nFbr + TbPerSlc - 1 ) >> logOfTPS; 
		unsigned int fb_st = fbrPtr0[mappedSlc] + localBId * fbrPerTb ;
		unsigned int fb_end = fbrPtr0[mappedSlc] + (localBId + 1) * fbrPerTb ;

		for (int fbr = fb_st + workId; fbr < fb_end && fbr < fbrPtr0[mappedSlc+1]; fbr+=warpPerSlice){
			
			tmp_val = 0;
			unsigned int idx0 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];    
	        
	        for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

		        unsigned int idx1 = dInds2[x];                    

	            for(unsigned int r=laneId; r<R; r+=32) {
	                tmp_val += vals[x] * dU1[idx1 * R + r]; 
	            }
	        }     
	        for(unsigned int r=laneId; r<R; r+=32) { 
	        	tmp = tmp_val * dU2[idx2 * R + r] ;
	        	atomicAdd(&dU0[idx0 * R + r], tmp); 
	        }    
		} 
	}
}

// CUDA kernel call to do HCSR MTTKRP 
__global__ void mttkrp_MIHCSR_kernel_smllBin_all_atomic(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
	ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  
	unsigned int slc = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) 
	// unsigned int slcPerTb = 16/warpPerSlice;
	// unsigned int shSlc = slc & slcPerTb;
	DTYPE tmp_val;
		              	              
	if(slc < nSlices){ 	    

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx1 = dfbrIdx0[mappedSlc] ;//slc;
    	int fb_st = fbrPtr0[mappedSlc];
		int fb_end = fbrPtr0[mappedSlc+1];

		for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
			
			unsigned int idx2 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];  

            // for(unsigned int r=laneId; r<R; r+=32) 
            // 	tmp_val = dU1[idx1 * R + r] * dU2[idx2 * R + r] ; //1PR
	        
	        for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

		        unsigned int idx0 = dInds2[x];                    

	            for(unsigned int r=laneId; r<R; r+=32) {
	            	tmp_val =  vals[x] * dU1[idx1 * R + r] * dU2[idx2 * R + r] ;
	            	atomicAdd(&dU0[idx0 * R + r], tmp_val); //2MR
	                // atomicAdd(&dU0[idx0 * R + r], (tmp_val * vals[x]) ); 
	            }
	        }   	
		}
	}
}

// CUDA fbr atomic sing slcLikeFbr
__global__ void mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar(DTYPE * vals, ITYPE *fbrLikeSlcInds, ITYPE *dInds2, 
	ITYPE *fbrPtr0, ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nFibers, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
	unsigned int fbr = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
	DTYPE tmp = 0, tmp_val;
		              	              
	if(fbr < nFibers - 1){ 	    
		
		tmp_val = 0;
		unsigned int idx2 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];  
		unsigned int idx1 = fbrLikeSlcInds[fbr];//slc;  

		for(unsigned int r=laneId; r<R; r+=32) 
           	tmp = dU1[idx1 * R + r] * dU2[idx2 * R + r] ; //1PR
        
        for(unsigned int x = fbrPtr1[fbr] + workId; x < fbrPtr1[fbr+1]; x+=warpPerSlice) {

	        unsigned int idx0 = dInds2[x];                    

            for(unsigned int r=laneId; r<R; r+=32) {
                tmp_val = vals[x] * tmp;///dU1[idx1 * R + r] * dU2[idx2 * R + r] ; //2MR
                atomicAdd(&dU0[idx0 * R + r], tmp_val);
            }
        }         
	}
}

// CUDA fbr atomic sing slcLikeFbr
__global__ void mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar_loop(DTYPE * vals, ITYPE *fbrLikeSlcInds, ITYPE *dInds2, 
	ITYPE *fbrPtr0, ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nFibers, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int warpId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //
	unsigned int blockId = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) //blockIdx.x ;//

	//like PARTI
	//hardcoded for 1 warp per nnz
	size_t num_loops_fbr = 1 * 32;
    size_t const fbr_per_loop = gridDim.x * blockDim.x;
    if(nFibers > fbr_per_loop) {
        num_loops_fbr = ((nFibers + fbr_per_loop - 1) / fbr_per_loop) << 5;
    }

	DTYPE tmp = 0, tmp_val;

	unsigned int fbr;

	for(size_t nl=0; nl<num_loops_fbr; ++nl) {
		
		fbr = (gId + nl * fbr_per_loop) >> 5;
		              	              
		if(fbr < nFibers - 1){ 	    
			
			tmp_val = 0;
			unsigned int idx2 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];  
			unsigned int idx1 = fbrLikeSlcInds[fbr];//slc;  

			for(unsigned int r=laneId; r<R; r+=32) 
	           	tmp = dU1[idx1 * R + r] * dU2[idx2 * R + r] ; //1PR
	        
	        for(unsigned int x = fbrPtr1[fbr] + warpId; x < fbrPtr1[fbr+1]; x+=warpPerSlice) {

		        unsigned int idx0 = dInds2[x];                    

	            for(unsigned int r=laneId; r<R; r+=32) {
	                tmp_val = vals[x] * tmp;///dU1[idx1 * R + r] * dU2[idx2 * R + r] ; //2MR
	                atomicAdd(&dU0[idx0 * R + r], tmp_val);
	            }
	        }    
		}
	}
}

// CUDA kernel call to do HCSR MTTKRP 
__global__ void mttkrp_MIHCSR_kernel_hvyBin_all_atomic(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
	ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int logOfTPS){
	
	unsigned int laneId = threadIdx.x & 31;
	unsigned int workId = threadIdx.x >> 5;
	unsigned int slc = blockIdx.x >> logOfTPS;
	unsigned int localBId = blockIdx.x & (TbPerSlc -1);
	
	DTYPE tmp = 0, tmp_val;
		              	              
	if(slc < nSlices){

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx1 = dfbrIdx0[mappedSlc] ;//slc;
		unsigned int nFbr = fbrPtr0[mappedSlc+1] - fbrPtr0[mappedSlc];		
		unsigned int fbrPerTb = (nFbr + TbPerSlc - 1 ) >> logOfTPS; 
		unsigned int fb_st = fbrPtr0[mappedSlc] + localBId * fbrPerTb ;
		unsigned int fb_end = fbrPtr0[mappedSlc] + (localBId + 1) * fbrPerTb ;

		for (int fbr = fb_st + workId; fbr < fb_end && fbr < fbrPtr0[mappedSlc+1]; fbr+=warpPerSlice){
			
			tmp_val = 0;
			unsigned int idx2 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]]; 

			for(unsigned int r=laneId; r<R; r+=32) 
            	tmp_val = dU1[idx1 * R + r] * dU2[idx2 * R + r] ;  
	        
	        for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

		        unsigned int idx0 = dInds2[x];                    

	            for(unsigned int r=laneId; r<R; r+=32) {
	            	// atomicAdd(&dU0[idx0 * R + r], (tmp_val * vals[x]) ); 
	            	tmp_val =  vals[x] * dU1[idx1 * R + r] * dU2[idx2 * R + r] ;
	                atomicAdd(&dU0[idx0 * R + r], tmp_val); 
	            }
	        }    
		} 
	}
}


int MTTKRP_COO_GPU(const Tensor &X, Matrix *U, const Options Opt){
	//allocate and memcpy GPU memory

	//Tensor
	ITYPE mode = Opt.mode;
	ITYPE R = Opt.R;
	ITYPE *dInds0, *dInds1, *dInds2, *dInds3;
	DTYPE *dVals;

	ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];

	checkCuda(cudaMalloc((void**) &dVals, X.totNnz * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dInds0, X.totNnz * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dInds1, X.totNnz * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dInds2, X.totNnz * sizeof(ITYPE)), 0);

	checkCuda(cudaMemcpy(dVals, &(X.vals[0]), X.totNnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dInds0, &(X.inds[mode0][0]), X.totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dInds1, &(X.inds[mode1][0]), X.totNnz * sizeof(ITYPE) ,cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dInds2, &(X.inds[mode2][0]), X.totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);

	// //Matrices
	DTYPE *dU0, *dU1, *dU2, *dU3;	
	checkCuda(cudaMalloc((void**) &dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU1, U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU2, U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE)), 0);

	cudaMemset(dU0, 0,  U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
	checkCuda(cudaMemcpy(dU1, &(U[mode1].vals[0]), U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dU2, &(U[mode2].vals[0]), U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);

	if(X.ndims == 4){
		ITYPE mode3 = X.modeOrder[3];
		checkCuda(cudaMalloc((void**) &dInds3, X.totNnz * sizeof(ITYPE)), 0);
		checkCuda(cudaMemcpy(dInds3, &(X.inds[mode3][0]), X.totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMalloc((void**) &dU3, U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE)), 0);
		checkCuda(cudaMemcpy(dU3, &(U[mode3].vals[0]), U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	}
	
	// BLOCK and GRID
	int BLOCKSIZE = 128;
	dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float mili = 0;
    bool useLoop = true;
	
	// /* Like PARTI loop */ = 
	if(useLoop)
		grid.x = 32768;
	else 
		grid.x = (32 * X.totNnz + BLOCKSIZE - 1) / BLOCKSIZE;
	
	// CUDA call
	cuda_timer_start(start);

	if(!useLoop){

		if(X.ndims == 3)
			mttkrp_COO_kernel<<<grid, block>>>(dVals, dInds0, dInds1, dInds2, X.totNnz, dU0, dU1, dU2, mode, R); 
		
		else if(X.ndims == 4)
			mttkrp_COO_kernel_4D<<<grid, block>>>(dVals, dInds0, dInds1, dInds2, dInds3, X.totNnz, dU0, dU1, dU2, dU3, mode, R); 
	
	}
	// /* loop like ParTI */
	else{

		if(X.ndims == 3)
			mttkrp_COO_kernel_loop<<<grid, block>>>(dVals, dInds0, dInds1, dInds2, X.totNnz, dU0, dU1, dU2, mode, R ); 
		
		else if(X.ndims == 4)
			mttkrp_COO_kernel_4D_loop<<<grid, block>>>(dVals, dInds0, dInds1, dInds2, dInds3, X.totNnz, dU0, dU1, dU2, dU3, mode, R); 
	
	}
	cuda_timer_stop(start, stop, mili);

	if(useLoop) cout << "Loop on. ";
    cout << "COO GPU using loop - time " << mili << "ms"<< endl;

	// check correctness
	checkCuda(cudaMemcpy(&U[mode0].vals[0], dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE), cudaMemcpyDeviceToHost), 0);
	// print_output(U, 0);
	cudaFree(dVals); 
	cudaFree(dU0); cudaFree(dU1); cudaFree(dU2); cudaFree(dU3);
	cudaFree(dInds0); cudaFree(dInds1); cudaFree(dInds2); cudaFree(dInds3);


	return 0;
}

int MTTKRP_HCSR_GPU(Tensor &X, Matrix *U, const Options &Opt){
	//allocate and memcpy GPU memory
	cout << "FIX fiber idx" << endl;
	//Tensor
	ITYPE *dInds2, *dInds3, *dfbrPtr0, *dfbrIdx0, *dfbrPtr1, *dfbrIdx1, *dFbrPtr2, *dFbrIdx2, *dSlcMapperBin;
	DTYPE *dVals;
	int logOfWarpPerSlice = log2(Opt.warpPerSlice);
	int TbPerSlc = 1;
	int logOfTPS = log2(TbPerSlc);

	ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];

	// dummy bin mapper to be compatible with bin mapper when bin are not used
	X.slcMapperBin.push_back(std::vector<ITYPE>());      
	for (int s = 0; s < X.fbrIdx[0].size(); ++s)
		X.slcMapperBin[0].push_back(s);

	checkCuda(cudaMalloc((void**) &dVals, X.totNnz * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dSlcMapperBin, X.slcMapperBin[0].size() * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dfbrIdx0, X.fbrIdx[0].size() * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dfbrPtr0, X.fbrPtr[0].size() * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dfbrPtr1, X.fbrPtr[1].size() * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dfbrIdx1, X.fbrIdx[1].size() * sizeof(ITYPE)), 0);

	checkCuda(cudaMemcpy(dVals, &(X.vals[0]), X.totNnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dSlcMapperBin, &(X.slcMapperBin[0][0]), X.slcMapperBin[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dfbrPtr0, &(X.fbrPtr[0][0]), X.fbrPtr[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dfbrIdx0, &(X.fbrIdx[0][0]), X.fbrIdx[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dfbrPtr1, &(X.fbrPtr[1][0]), X.fbrPtr[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dfbrIdx1, &(X.fbrIdx[1][0]), X.fbrIdx[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);

	// //Matrices
	DTYPE *dU0, *dU1, *dU2, *dU3;	
	checkCuda(cudaMalloc((void**) &dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU1, U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU2, U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE)), 0);

	cudaMemset(dU0, 0,  U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
	checkCuda(cudaMemcpy(dU1, &(U[mode1].vals[0]), U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dU2, &(U[mode2].vals[0]), U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	
	if(X.ndims == 3){
		checkCuda(cudaMalloc((void**) &dInds2, X.totNnz * sizeof(ITYPE)), 0);
		checkCuda(cudaMemcpy(dInds2, &(X.inds[mode2][0]), X.totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	}

	if(X.ndims == 4){
		ITYPE mode3 = X.modeOrder[3];
		checkCuda(cudaMalloc((void**) &dFbrIdx2, X.fbrIdx[2].size() * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dFbrPtr2, X.fbrPtr[2].size() * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dInds3, X.totNnz * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dU3, U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE)), 0);
		
		checkCuda(cudaMemcpy(dFbrPtr2, &(X.fbrPtr[2][0]), X.fbrPtr[2].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dFbrIdx2, &(X.fbrIdx[2][0]), X.fbrIdx[2].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dInds3, &(X.inds[mode3][0]), X.totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dU3, &(U[mode3].vals[0]), U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	}

	// BLOCK and GRID
	int BLOCKSIZE = 512;

	if(Opt.warpPerSlice * 32 > BLOCKSIZE){
		cout << "BLOCKSIZE is smaller than work per slice! Increase BLOCKSIZE." << endl;
		exit(0);
	}

	dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);
	grid.x = (Opt.warpPerSlice * 32 * X.dims[mode0] + BLOCKSIZE - 1) / BLOCKSIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float mili = 0;

	checkCuda(cudaEventRecord(start), __LINE__);

	// mttkrp_HCSR_kernel_COO<<<grid, block, 32 * sizeof(DTYPE)>>>(dVals, dfbrIdx0, dSlcMapperBin, dInds2, dfbrPtr0, dfbrPtr1, dfbrIdx1,
	// 	X.fbrIdx[0].size(), dU0, dU1, dU2,Opt.mode, Opt.R, Opt.warpPerSlice, logOfWarpPerSlice, TbPerSlc, logOfTPS); 
	if(X.ndims == 3)
		mttkrp_HCSR_kernel_smllBin<<<grid, block, 32 * sizeof(DTYPE)>>>(dVals, dfbrIdx0, dSlcMapperBin, dInds2, dfbrPtr0, dfbrPtr1, dfbrIdx1,
		X.fbrIdx[0].size(), dU0, dU1, dU2,Opt.mode, Opt.R, Opt.warpPerSlice, logOfWarpPerSlice, TbPerSlc, logOfTPS); 
	else
		mttkrp_HCSR_kernel_smllBin_4D<<<grid, block, 32 * sizeof(DTYPE)>>>(dVals, dfbrIdx0, dSlcMapperBin, dInds3, dfbrPtr0, dfbrPtr1, dfbrIdx1,
		dFbrPtr2, dFbrIdx2, X.fbrIdx[0].size(), dU0, dU1, dU2, dU3, Opt.mode, Opt.R, Opt.warpPerSlice, logOfWarpPerSlice, TbPerSlc, logOfTPS); 


	checkCuda(cudaEventRecord(stop), __LINE__);
    cudaEventSynchronize(stop);
    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
    cudaDeviceSynchronize();
    cout << "HCSR GPU - time " << mili << "ms"<< endl;

	// check correctness
	checkCuda(cudaMemcpy(&U[mode0].vals[0], dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE), cudaMemcpyDeviceToHost), 0);
	cudaFree(dVals); 
	cudaFree(dU0); cudaFree(dU1); cudaFree(dU2); cudaFree(dU3);
	cudaFree(dInds2); cudaFree(dInds3); 
	cudaFree(dfbrIdx0); cudaFree(dfbrIdx1); cudaFree(dFbrIdx2);
	cudaFree(dfbrPtr0); cudaFree(dfbrPtr1); cudaFree(dFbrPtr2);

	return 0;
}

int MTTKRP_TILED_COO_GPU(TiledTensor *TiledX, Matrix *U, const Options Opt){
	//allocate and memcpy GPU memory

	//Tensor
	ITYPE mode = Opt.mode;
	ITYPE R = Opt.R;
	ITYPE *dInds0, *dInds1, *dInds2;
	ITYPE dLoc = 0, totNnz = 0;
	DTYPE *dVals;

	// All tile same mode
	ITYPE mode0 = TiledX[0].modeOrder[0];
    ITYPE mode1 = TiledX[0].modeOrder[1];
    ITYPE mode2 = TiledX[0].modeOrder[2];

	for (int tile = 0; tile < Opt.nTile; ++tile)
		totNnz += TiledX[tile].totNnz;

	checkCuda(cudaMalloc((void**) &dVals, totNnz * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dInds0, totNnz * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dInds1, totNnz * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dInds2, totNnz * sizeof(ITYPE)), 0);

	for (int tile = 0; tile < Opt.nTile; ++tile){
		
		if(tile > 0) 
			dLoc += TiledX[tile-1].totNnz;

		checkCuda(cudaMemcpy(dVals + dLoc, &(TiledX[tile].vals[0]), TiledX[tile].totNnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dInds0 + dLoc, &(TiledX[tile].inds[mode0][0]), TiledX[tile].totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dInds1 + dLoc, &(TiledX[tile].inds[mode1][0]), TiledX[tile].totNnz * sizeof(ITYPE) ,cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dInds2 + dLoc, &(TiledX[tile].inds[mode2][0]), TiledX[tile].totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	}

	// //Matrices
	DTYPE *dU0, *dU1, *dU2;	
	checkCuda(cudaMalloc((void**) &dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU1, U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU2, U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE)), 0);

	cudaMemset(dU0, 0,  U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
	checkCuda(cudaMemcpy(dU1, &(U[mode1].vals[0]), U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dU2, &(U[mode2].vals[0]), U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	
	// BLOCK and GRID
	int BLOCKSIZE = 128;
	dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);
	
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float mili = 0, GPUTime = 0;

	// CUDA call
	dLoc = 0;
	for (int tile = 0; tile < Opt.nTile; ++tile){
		
		if(tile > 0) 
			dLoc += TiledX[tile-1].totNnz;

		cout << "Tile " << tile << " launched.. "<<endl;
		
		grid.x = (32 * TiledX[tile].totNnz + BLOCKSIZE - 1) / BLOCKSIZE;

		checkCuda(cudaEventRecord(start), __LINE__);
		mttkrp_COO_kernel<<<grid, block>>>(dVals + dLoc, dInds0 + dLoc, dInds1 + dLoc, dInds2 + dLoc, TiledX[tile].totNnz, dU0, dU1, dU2,
								mode, R); 
	
		checkCuda(cudaEventRecord(stop), __LINE__);
	    cudaEventSynchronize(stop);
	    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
	    cudaDeviceSynchronize();
	    cout << "Tile: " << tile << " - time " << mili << "ms"<< endl;
	    GPUTime += mili;
	   
	}
	cout << "COO GPU - time " << GPUTime << "ms"<< endl;

	// check correctness
	checkCuda(cudaMemcpy(&U[mode0].vals[0], dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE), cudaMemcpyDeviceToHost), 0);
	cudaFree(dVals); 
	cudaFree(dU0); cudaFree(dU1); cudaFree(dU2);
	cudaFree(dInds0); cudaFree(dInds1); cudaFree(dInds2);

	return 0;
}

int MTTKRP_B_HCSR_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt){
	
	/* Allocate and memcpy GPU memory */
	//Tensor
	ITYPE *dInds2, *dInds3, *dfbrPtr0, *dfbrIdx0, *dfbrPtr1, *dfbrIdx1, *dFbrPtr2, *dFbrIdx2, *dSlcMapperBin, *dFbrLikeSlcInds;
	DTYPE *dVals;
	ITYPE dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0, dFbrLoc =0,  dFbrIdxLoc =0, dBinLoc = 0, dFbrLoc2 =0;
	ITYPE totNnz = 0, totSlcPtr = 0, totSlcIdx = 0, totFbrPtr = 0, totFbrIdx = 0, totFbrPtr2 = 0;

	// // All tile same mode
	ITYPE mode0 = TiledX[0].modeOrder[0];
    ITYPE mode1 = TiledX[0].modeOrder[1];
    ITYPE mode2 = TiledX[0].modeOrder[2];
    ITYPE mode3 =((TiledX[0].ndims == 4) ? TiledX[0].modeOrder[3] : 0) ;

	for (int tile = 0; tile < Opt.nTile; ++tile){
		totNnz += TiledX[tile].totNnz;
		totSlcPtr += TiledX[tile].fbrPtr[0].size() ;
		totSlcIdx += TiledX[tile].fbrIdx[0].size() ;
		totFbrPtr += TiledX[tile].fbrPtr[1].size() ;
		totFbrIdx += TiledX[tile].fbrIdx[1].size() ;
		totFbrPtr2 += ((TiledX[tile].ndims == 4) ? TiledX[tile].fbrPtr[2].size() : 0) ;
	}

	double t0 = seconds();
	checkCuda(cudaMalloc((void**) &dVals, totNnz * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dfbrPtr0, totSlcPtr * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dfbrIdx0, totSlcIdx * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dSlcMapperBin, totSlcPtr * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dfbrPtr1, totFbrPtr * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dfbrIdx1, totFbrIdx * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dFbrLikeSlcInds, totFbrIdx * sizeof(ITYPE)), 0);

	if(TiledX[0].ndims == 3)
		checkCuda(cudaMalloc((void**) &dInds2, totNnz * sizeof(ITYPE)), 0);

	if(TiledX[0].ndims == 4){
		checkCuda(cudaMalloc((void**) &dFbrIdx2, totFbrPtr2 * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dFbrPtr2, totFbrPtr2 * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dInds3, totNnz * sizeof(ITYPE)), 0);
	}

	/* cuda memcopy for tiled parts*/
	for (int tile = 0; tile < Opt.nTile; ++tile){	
		if(tile > 0) {
			dLoc += TiledX[tile-1].totNnz;
			dSlcLoc += TiledX[tile - 1].fbrPtr[0].size(); // all tile same
			dSlcIdxLoc += TiledX[tile - 1].fbrIdx[0].size(); 
			dFbrLoc += TiledX[tile - 1].fbrPtr[1].size();
			dFbrIdxLoc += TiledX[tile - 1].fbrIdx[1].size();
			dFbrLoc2 += ((TiledX[tile].ndims == 4) ? TiledX[tile - 1].fbrPtr[2].size() : 0) ;
		}

		checkCuda(cudaMemcpy(dVals + dLoc, &(TiledX[tile].vals[0]), TiledX[tile].totNnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dfbrPtr0 + dSlcLoc, &(TiledX[tile].fbrPtr[0][0]), TiledX[tile].fbrPtr[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dfbrIdx0 + dSlcIdxLoc, &(TiledX[tile].fbrIdx[0][0]), TiledX[tile].fbrIdx[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dfbrPtr1 + dFbrLoc, &(TiledX[tile].fbrPtr[1][0]), TiledX[tile].fbrPtr[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dfbrIdx1 + dFbrIdxLoc, &(TiledX[tile].fbrIdx[1][0]), TiledX[tile].fbrIdx[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		
		if(Opt.impType == 14)
			checkCuda(cudaMemcpy(dFbrLikeSlcInds + dFbrIdxLoc, &(TiledX[tile].fbrLikeSlcInds[0]), TiledX[tile].fbrIdx[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	
		if(TiledX[tile].ndims == 3)
			checkCuda(cudaMemcpy(dInds2 + dLoc, &(TiledX[tile].inds[TiledX[tile].modeOrder[2]][0]), TiledX[tile].totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);			

		if(TiledX[tile].ndims == 4){			
			checkCuda(cudaMemcpy(dFbrPtr2 + dFbrLoc2, &(TiledX[tile].fbrPtr[2][0]), TiledX[tile].fbrPtr[2].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
			checkCuda(cudaMemcpy(dFbrIdx2 + dFbrLoc2, &(TiledX[tile].fbrIdx[2][0]), TiledX[tile].fbrIdx[2].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
			checkCuda(cudaMemcpy(dInds3 + dLoc, &(TiledX[tile].inds[mode3][0]), TiledX[tile].totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		}

		dBinLoc = 0;
		for (int bin = 0; bin < Opt.nBin; ++bin){

			if(bin > 0)
				dBinLoc += TiledX[tile].slcMapperBin[bin-1].size();

		    checkCuda(cudaMemcpy(dSlcMapperBin + dSlcIdxLoc + dBinLoc, &(TiledX[tile].slcMapperBin[bin][0]), TiledX[tile].slcMapperBin[bin].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		}
	}
	float tnsMemcpyTime = seconds() - t0;

	t0 = seconds();
	// //Matrices
	DTYPE *dU0, *dU1, *dU2, *dU3;	
	checkCuda(cudaMalloc((void**) &dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU1, U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU2, U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE)), 0);

	cudaMemset(dU0, 0,  U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
	checkCuda(cudaMemcpy(dU1, &(U[mode1].vals[0]), U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dU2, &(U[mode2].vals[0]), U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);

	float mtxMemcpyTime = seconds() - t0;

	// cout << "tns and mtx memcopy time: " << tnsMemcpyTime <<", " << mtxMemcpyTime<< endl;
	
	if(TiledX[0].ndims == 4){
		checkCuda(cudaMalloc((void**) &dU3, U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE)), 0);
		checkCuda(cudaMemcpy(dU3, &(U[mode3].vals[0]), U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	}

	// BLOCK and GRID
	int BLOCKSIZE = 512;
	unsigned int rowInATB = BLOCKSIZE / (Opt.warpPerSlice*32); 

	if(Opt.warpPerSlice * 32 > BLOCKSIZE){
		cout << "BLOCKSIZE is smaller than work per slice! Increase BLOCKSIZE." << endl;
		exit(0);
	}

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStream_t streams[Opt.nBin];
    float mili = 0, GPUTime = 0, CPUtimer = 0, allModeGPUTime = 0;

    int smallBinEndsAt = 5;

    /* Warp per slice and threadblock per size */
    int *warpPerSlc = new int[Opt.nBin];
    int *logOfWarpPerSlc = new int[Opt.nBin];
    int *TbPerSlc = new int[Opt.nBin];
    int *logOfTbPerSlc = new int[Opt.nBin];

    for (int bin = 0; bin < Opt.nBin ; ++bin){
    	
    	TbPerSlc[bin] = 1;
		warpPerSlc[bin] = ((bin > 0) ? 2 << (bin - 1) : 1);
		
		if(warpPerSlc[bin] > 16)		
			warpPerSlc[bin] = 16;

		logOfWarpPerSlc[bin] = log2(warpPerSlc[bin]);

		TbPerSlc[bin] = 1;
		logOfTbPerSlc[bin] = 0;
		
		if (bin >= smallBinEndsAt){
		
			TbPerSlc[bin] = 1 << (bin - smallBinEndsAt + 1); // 1st big bin starts with 1 TB 1 << 1 not 1 << 5
			if(TbPerSlc[bin] > 32) TbPerSlc[bin] = 32;		
			logOfTbPerSlc[bin] = log2(TbPerSlc[bin]);

			warpPerSlc[bin] = 16;
			logOfWarpPerSlc[bin] = 4;
		}
    }

    // TBD: change warpPerSlc to warpPerSlc[bin] and all
	int slcPerTb = 1;

	dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0; dFbrLoc =0, dFbrIdxLoc = 0, dFbrLoc2= 0;

	for (int bin = 0; bin < Opt.nBin; ++bin)
		cudaStreamCreate(&streams[bin]);

	/*MTTKRP on Opt.mode*/
	int MTTKRPmode = mode0;//Opt.mode;
		
	for (int tile = 0; tile < Opt.nTile; ++tile){

		dBinLoc = 0;
		
		if(tile > 0) {
			dLoc += TiledX[tile-1].totNnz;
			dSlcLoc += TiledX[tile - 1].fbrPtr[0].size(); 
			dSlcIdxLoc += TiledX[tile - 1].fbrIdx[0].size(); 
			dFbrLoc += TiledX[tile - 1].fbrPtr[1].size();
			dFbrIdxLoc += TiledX[tile - 1].fbrIdx[1].size();
			dFbrLoc2 += ((TiledX[0].ndims == 4) ? TiledX[tile - 1].fbrPtr[2].size() : 0) ;
		}

		BLOCKSIZE = 512;
		dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

		int smallBinEndsAt = 5;
		int slcPerTb = 0;

		double t0 = seconds();
		cuda_timer_start(start);
		
		// Process small bins.. accepts 2 slice 1 TB
		for (int bin = 0; bin < Opt.nBin ; ++bin){

			if(bin < smallBinEndsAt){
				
				ITYPE shSize = 0;//slcPerTb * 32 * sizeof(DTYPE); slcPerTb = 16 / warpPerSlc[bin];

				dBinLoc += ((bin > 0) ? TiledX[tile].slcMapperBin[bin-1].size() : 0);

				grid.x = ( TbPerSlc[bin] * warpPerSlc[bin] * 32 * TiledX[tile].slcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;

				if(TiledX[0].ndims == 3)
					mttkrp_HCSR_kernel_smllBin<<<grid, block, shSize , streams[bin]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
					dInds2 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dfbrIdx1 + dFbrLoc, TiledX[tile].slcMapperBin[bin].size(), 
					dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlc[bin], logOfWarpPerSlc[bin], TbPerSlc[bin], logOfTbPerSlc[bin]); 
				else
					mttkrp_HCSR_kernel_smllBin_4D<<<grid, block, shSize , streams[bin]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
					dInds3 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc, dfbrIdx1 + dFbrIdxLoc, dFbrPtr2 + dFbrLoc2, dFbrIdx2 + dFbrLoc2, TiledX[tile].slcMapperBin[bin].size(), 
					dU0, dU1, dU2, dU3, Opt.mode, Opt.R, warpPerSlc[bin], logOfWarpPerSlc[bin], TbPerSlc[bin], logOfTbPerSlc[bin]); 
			}
			
			// Processing heavy bin.. multiple TB per slice
			else{

				dBinLoc += TiledX[tile].slcMapperBin[bin-1].size();
						
				grid.x = (TbPerSlc[bin] * warpPerSlc[bin] * 32 * TiledX[tile].slcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;
				
					if(TiledX[0].ndims == 3)
						mttkrp_HCSR_kernel_hvyBin<<<grid, block, 0, streams[bin]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
						dInds2 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dfbrIdx1 + dFbrLoc, TiledX[tile].slcMapperBin[bin].size(), 
						dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlc[bin], logOfWarpPerSlc[bin],  TbPerSlc[bin], logOfTbPerSlc[bin]); 
					else
						mttkrp_HCSR_kernel_hvyBin_4D<<<grid, block, 0, streams[bin]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
						dInds3 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc, dfbrIdx1 + dFbrIdxLoc, dFbrPtr2 + dFbrLoc2, dFbrIdx2 + dFbrLoc2, TiledX[tile].slcMapperBin[bin].size(), 
						dU0, dU1, dU2, dU3, Opt.mode, Opt.R, warpPerSlc[bin], logOfWarpPerSlc[bin],  TbPerSlc[bin], logOfTbPerSlc[bin]); 
			}
		}
	
		cuda_timer_stop(start, stop, mili);
	    CPUtimer += seconds() - t0;
	    GPUTime += mili;

	    if(Opt.verbose){
	    	cout << "Tile: " << tile << " - time: " << mili << "ms";
	    	cout <<" nnz: " << TiledX[tile].totNnz << " nFibers: "
	    	<< TiledX[tile].fbrPtr[1].size() << " nSlc " << TiledX[tile].fbrIdx[0].size() << " ";
			cout << endl;
		} 
	}
	
	allModeGPUTime += GPUTime;
	cout << "B-CSF-GPU-mode " << MTTKRPmode <<" :" << GPUTime << "," << endl;

	for (int bin = 0; bin < Opt.nBin; ++bin)
		cudaStreamDestroy(streams[bin]);

	// check correctness
	checkCuda(cudaMemcpy(&U[mode0].vals[0], dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE), cudaMemcpyDeviceToHost), 0);

	cudaFree(dVals); 
	cudaFree(dU0); cudaFree(dU1); cudaFree(dU2); cudaFree(dU3);
	cudaFree(dfbrIdx0); cudaFree(dInds2); cudaFree(dInds3); 
	cudaFree(dfbrIdx0); cudaFree(dfbrIdx1); cudaFree(dFbrIdx2);
	cudaFree(dfbrPtr0); cudaFree(dfbrPtr1); cudaFree(dFbrPtr2);
	cudaFree(dFbrLikeSlcInds);

	return 0;
}

int MTTKRP_HYB_GPU(const HYBTensor &HybX, Matrix *U, const Options &Opt){
	//allocate and memcpy GPU memory

	//Tensor
	ITYPE *dCOOInds0, *dCOOInds1, *dCOOInds2, *dCOOInds3;
	ITYPE *dCSLSlcPtr, *dCSLSlcInds, *dCSLInds1, *dCSLInds2, *dCSLSlcMapperBin;
	ITYPE *dfbrPtr0, *dfbrIdx0, *dInds2, *dInds3, *dfbrPtr1, *dfbrIdx1,  *dFbrPtr2, *dFbrIdx2, *dSlcMapperBin;

	DTYPE *dVals, *dCOOVals, *dCSLVals;
	ITYPE dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0, dFbrLoc =0, dFbrIdxLoc =0, dBinLoc = 0, dCSLBinLoc = 0, dFbrLoc2 =0;
	int warpPerSlice = Opt.warpPerSlice;
	int logOfWarpPerSlice = log2(Opt.warpPerSlice);
	int TbPerSlc = 1;
	int logOfTPS = log2(TbPerSlc);

	// All tile same mode
	ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];
    ITYPE mode3 =((HybX.ndims == 4) ? HybX.modeOrder[3] : 0) ;

    // ****** mem op HYB COO *******
    if(HybX.COOnnz > 0){
		
		checkCuda(cudaMalloc((void**) &dCOOVals, HybX.COOnnz * sizeof(DTYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCOOInds0, HybX.COOnnz * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCOOInds1, HybX.COOnnz * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCOOInds2, HybX.COOnnz * sizeof(ITYPE)), 0);

		checkCuda(cudaMemcpy(dCOOVals, &(HybX.COOvals[0]), HybX.COOnnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dCOOInds0, &(HybX.COOinds[mode0][0]), HybX.COOnnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dCOOInds1, &(HybX.COOinds[mode1][0]), HybX.COOnnz * sizeof(ITYPE) ,cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dCOOInds2, &(HybX.COOinds[mode2][0]), HybX.COOnnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		
		if(HybX.ndims == 4){
			checkCuda(cudaMalloc((void**) &dCOOInds3, HybX.COOnnz * sizeof(ITYPE)), 0);
			checkCuda(cudaMemcpy(dCOOInds3, &(HybX.COOinds[mode3][0]), HybX.COOnnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		}
	}

   // ****** mem op HYB CSL *******

	if(HybX.CSLnnz > 0){

		checkCuda(cudaMalloc((void**) &dCSLVals, HybX.CSLnnz * sizeof(DTYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCSLSlcPtr,  HybX.CSLslicePtr.size() * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCSLSlcInds, HybX.CSLsliceIdx.size() * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCSLInds1, HybX.CSLnnz * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCSLInds2, HybX.CSLnnz * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCSLSlcMapperBin, HybX.CSLslicePtr.size() * sizeof(ITYPE)), 0);

		checkCuda(cudaMemcpy(dCSLVals, &(HybX.CSLvals[0]), HybX.CSLnnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);	
		checkCuda(cudaMemcpy(dCSLSlcPtr + dSlcLoc, &(HybX.CSLslicePtr[0]), HybX.CSLslicePtr.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dCSLSlcInds + dSlcIdxLoc, &(HybX.CSLsliceIdx[0]), HybX.CSLsliceIdx.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dCSLInds1, &(HybX.CSLinds[mode1][0]), HybX.CSLnnz * sizeof(ITYPE) ,cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dCSLInds2, &(HybX.CSLinds[mode2][0]), HybX.CSLnnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		
		dCSLBinLoc = 0;
		for (int bin = 0; bin < Opt.nBin; ++bin){

			if(bin > 0)
				dCSLBinLoc += HybX.CSLslcMapperBin[bin-1].size();

			if(HybX.CSLslcMapperBin[bin].size() > 0)
		    	checkCuda(cudaMemcpy(dCSLSlcMapperBin + dSlcIdxLoc + dCSLBinLoc, &(HybX.CSLslcMapperBin[bin][0]), HybX.CSLslcMapperBin[bin].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		}
	}
 
    // ****** mem op HYB HCSR *******

    if(HybX.HCSRnnz > 0){

		checkCuda(cudaMalloc((void**) &dVals, HybX.HCSRnnz * sizeof(DTYPE)), 0);
		checkCuda(cudaMalloc((void**) &dfbrPtr0,  HybX.fbrPtr[0].size() * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dfbrIdx0, HybX.fbrIdx[0].size() * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dSlcMapperBin, HybX.fbrPtr[0].size() * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dfbrPtr1, HybX.fbrPtr[1].size()  * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dfbrIdx1, HybX.fbrPtr[1].size() * sizeof(ITYPE)), 0);

		checkCuda(cudaMemcpy(dVals, &(HybX.vals[0]), HybX.HCSRnnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dfbrPtr0, &(HybX.fbrPtr[0][0]), HybX.fbrPtr[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dfbrIdx0, &(HybX.fbrIdx[0][0]), HybX.fbrIdx[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dfbrPtr1, &(HybX.fbrPtr[1][0]), HybX.fbrPtr[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dfbrIdx1, &(HybX.fbrIdx[1][0]), HybX.fbrPtr[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);

	    if(HybX.ndims == 3){
	    	checkCuda(cudaMalloc((void**) &dInds2, HybX.HCSRnnz * sizeof(ITYPE)), 0);
	    	checkCuda(cudaMemcpy(dInds2, &(HybX.inds[mode2][0]), HybX.HCSRnnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		}

	    if(HybX.ndims == 4){
	        checkCuda(cudaMalloc((void**) &dFbrIdx2, HybX.fbrIdx[2].size() * sizeof(ITYPE)), 0);
	        checkCuda(cudaMalloc((void**) &dFbrPtr2, HybX.fbrPtr[2].size() * sizeof(ITYPE)), 0);
	        checkCuda(cudaMalloc((void**) &dInds3, HybX.HCSRnnz * sizeof(ITYPE)), 0);
	        checkCuda(cudaMemcpy(dFbrPtr2, &(HybX.fbrPtr[2][0]), HybX.fbrPtr[2].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
			checkCuda(cudaMemcpy(dFbrIdx2, &(HybX.fbrIdx[2][0]), HybX.fbrIdx[2].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
			checkCuda(cudaMemcpy(dInds3, &(HybX.inds[mode3][0]), HybX.HCSRnnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	    }

		dBinLoc = 0;
		for (int bin = 0; bin < Opt.nBin; ++bin){

			if(bin > 0)
				dBinLoc += HybX.slcMapperBin[bin-1].size();

			if(HybX.slcMapperBin[bin].size() > 0)
		    	checkCuda(cudaMemcpy(dSlcMapperBin + dSlcIdxLoc + dBinLoc, &(HybX.slcMapperBin[bin][0]), HybX.slcMapperBin[bin].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		}
	}
	
	// //Matrices
	DTYPE *dU0, *dU1, *dU2, *dU3;	
	checkCuda(cudaMalloc((void**) &dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU1, U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU2, U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE)), 0);

	cudaMemset(dU0, 0,  U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
	checkCuda(cudaMemcpy(dU1, &(U[mode1].vals[0]), U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dU2, &(U[mode2].vals[0]), U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);

	if(HybX.ndims == 4){
        checkCuda(cudaMalloc((void**) &dU3, U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE)), 0);
        checkCuda(cudaMemcpy(dU3, &(U[mode3].vals[0]), U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
    }
	
	// BLOCK and GRID
	int BLOCKSIZE = 512;
	dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);
	unsigned int rowInATB = BLOCKSIZE / (Opt.warpPerSlice*32); 

	if(Opt.warpPerSlice * 32 > BLOCKSIZE){
		cout << "BLOCKSIZE is smaller than work per slice! Increase BLOCKSIZE." << endl;
		exit(0);
	}

    cudaEvent_t start, stop, HYBstart, HYBstop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&HYBstart);
    cudaEventCreate(&HYBstop);

    cudaStream_t streams[2 * Opt.nBin + 1];
	for (int bin = 0; bin < 2 * Opt.nBin + 1; ++bin)
		cudaStreamCreate(&streams[bin]);

    float mili = 0, HYBmili =0, GPUTime = 0, CPUtimer = 0, HYBTime = 0;
	dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0; dFbrLoc =0;
	bool useLoop = false;
	
	if(useLoop)
		grid.x = 32768*2;

			// mili = 0; 
	dCSLBinLoc = 0; dBinLoc = 0;

	int smallBinEndsAt = 5;
	int slcPerTb = 0;

	cuda_timer_start(HYBstart);

	// ******* CUDA COO *******

	// if(HybX.COOnnz > 0){

	// 	BLOCKSIZE = 128;
	// 	block.x = BLOCKSIZE;
	// 		// /* Like PARTI loop */ = 

	// 	if(!useLoop)
	// 		grid.x = (32 * HybX.COOnnz + BLOCKSIZE - 1) / BLOCKSIZE;

	// 	if(Opt.verbose) 
	// 		cuda_timer_start(start);
  		
 //  		if(!useLoop){

	//   		if(HybX.ndims == 3)
	// 			mttkrp_HYB_COO_kernel<<<grid, block, 0, 0>>>(dCOOVals, dCOOInds0, dCOOInds1, dCOOInds2, HybX.COOnnz, dU0, dU1, dU2,	Opt.mode, Opt.R); 
	// 		else if (HybX.ndims == 4)
	// 			mttkrp_HYB_COO_kernel_4D<<<grid, block, 0, 0>>>(dCOOVals, dCOOInds0, dCOOInds1, dCOOInds2,dCOOInds3, HybX.COOnnz, dU0, dU1, dU2, dU3, Opt.mode, Opt.R); 
	// 	}

	// 	else{
  			
	//   		if(HybX.ndims == 3)
	// 			mttkrp_HYB_COO_kernel_loop<<<grid, block, 0, 0>>>(dCOOVals, dCOOInds0, dCOOInds1, dCOOInds2, HybX.COOnnz, dU0, dU1, dU2,	Opt.mode, Opt.R); 
	// 		else if (HybX.ndims == 4)
	// 			mttkrp_HYB_COO_kernel_4D_loop<<<grid, block, 0, 0>>>(dCOOVals, dCOOInds0, dCOOInds1, dCOOInds2,dCOOInds3, HybX.COOnnz, dU0, dU1, dU2, dU3, Opt.mode, Opt.R); 
	// 	}

	//     if(Opt.verbose){
	//     	cuda_timer_stop(start, stop, mili);
	//     	HYBTime += mili;
	//     	cout << "HYB-COO GPU " << mili << "ms"<< endl;
	//     }
	// }
	// ******* CUDA CSL *******

	// if(HybX.CSLnnz > 0 || HybX.HCSRnnz > 0)
	{
		if(HybX.COOnnz > 0){

			BLOCKSIZE = 128;
			block.x = 128;
			grid.x = (32 * HybX.COOnnz + BLOCKSIZE - 1) / BLOCKSIZE;

	  		if(HybX.ndims == 3)
				mttkrp_HYB_COO_kernel<<<grid, block, 0, 0>>>(dCOOVals, dCOOInds0, dCOOInds1, dCOOInds2, HybX.COOnnz, dU0, dU1, dU2,	Opt.mode, Opt.R); 
			else if (HybX.ndims == 4)
				mttkrp_HYB_COO_kernel_4D<<<grid, block, 0, 0>>>(dCOOVals, dCOOInds0, dCOOInds1, dCOOInds2,dCOOInds3, HybX.COOnnz, dU0, dU1, dU2, dU3, Opt.mode, Opt.R); 
		
		}

		BLOCKSIZE = 512;
		block.x = BLOCKSIZE;

		for (int bin = 0; bin < Opt.nBin ; ++bin){

			dBinLoc += ((bin > 0) ? HybX.slcMapperBin[bin-1].size() : 0);
			dCSLBinLoc += ((bin > 0) ? HybX.CSLslcMapperBin[bin-1].size() : 0);

			if( HybX.slcMapperBin[bin].size() == 0 && HybX.CSLslcMapperBin[bin].size() == 0)
				continue;
			// Processing small bin.. merged to one. 1 WARP slice
			if(bin < smallBinEndsAt){

				warpPerSlice = 1;
				logOfWarpPerSlice = 0;//log2(warpPerSlice);
				slcPerTb = 16 / warpPerSlice;

				/* CSL small bin */
				if(HybX.CSLnnz > 0){

					grid.x = ( warpPerSlice * 32 * HybX.CSLslcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;

					mttkrp_CSL_kernel_bin<<<grid, block, 0, streams[1]>>>(dCSLVals, dCSLSlcInds, dCSLSlcMapperBin + dCSLBinLoc, 
						dCSLInds2, dCSLSlcPtr, dCSLInds1, HybX.CSLslcMapperBin[bin].size(), 
						dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice); 
				}
				
				/* HCSR small bin */
				if(HybX.HCSRnnz > 0){

					grid.x = ( warpPerSlice * 32 * HybX.slcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;

					if(HybX.ndims == 3)
						mttkrp_HCSR_kernel_smllBin<<<grid, block, 0, streams[2]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
						dInds2 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dfbrIdx1 + dFbrLoc, HybX.slcMapperBin[bin].size(), 
						dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice, TbPerSlc, logOfTPS); 
					
					else if(HybX.ndims == 4)
						mttkrp_HCSR_kernel_smllBin_4D<<<grid, block, 0, streams[2]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
						dInds3 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dfbrIdx1 + dFbrLoc, dFbrPtr2 + dFbrLoc2, dFbrIdx2 + dFbrLoc2, HybX.slcMapperBin[bin].size(), 
						dU0, dU1, dU2, dU3, Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice, TbPerSlc, logOfTPS); 
				}
			}

			// Processing heavy bin.. multiple TB per slice
			else{
		
				TbPerSlc = 1 << (bin - smallBinEndsAt + 1); // 1st big bin starts with 1 TB 1 << 1 not 1 << 5

				if(TbPerSlc > 32) TbPerSlc = 32;		
				logOfTPS = log2(TbPerSlc);

				warpPerSlice = 16;
				logOfWarpPerSlice = 4;

				/* CSL big bin */
				if(HybX.CSLnnz > 0){	
					grid.x = (TbPerSlc * warpPerSlice * 32 * HybX.CSLslcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;
					
					mttkrp_CSL_kernel_hvyBin<<<grid, block, 0, streams[bin+1]>>>(dCSLVals + dLoc, dCSLSlcInds + dSlcIdxLoc, dCSLSlcMapperBin + dSlcIdxLoc + dCSLBinLoc, 
						dCSLInds2 + dLoc, dCSLSlcPtr + dSlcLoc, dCSLInds1, HybX.CSLslcMapperBin[bin].size(), 
						dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice,  TbPerSlc, logOfTPS); 
				}

				/* HCSR big bin */
				if(HybX.HCSRnnz > 0){
					grid.x = (TbPerSlc * warpPerSlice * 32 * HybX.slcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;
					
					if(HybX.ndims == 3)
						mttkrp_HCSR_kernel_hvyBin<<<grid, block, 0, streams[bin+2]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
							dInds2 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dfbrIdx1 + dFbrLoc, HybX.slcMapperBin[bin].size(), 
							dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice,  TbPerSlc, logOfTPS); 
						
					else if(HybX.ndims == 4)
	                    mttkrp_HCSR_kernel_hvyBin_4D<<<grid, block, 0, streams[bin + 2]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
	                    dInds3 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc, dfbrIdx1 + dFbrIdxLoc, dFbrPtr2 + dFbrLoc2, dFbrIdx2 + dFbrLoc2, HybX.slcMapperBin[bin].size(), 
	                    dU0, dU1, dU2, dU3, Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice,  TbPerSlc, logOfTPS);
		        } 	

			}
		}

	    // if(Opt.verbose){
	    // 	cuda_timer_stop(start, stop, mili);
	    // 	HYBTime += mili;
	    // 	cout << "CSL+HCSR GPU-time: " << mili << "ms"<< endl;
	    // }
	}

	cuda_timer_stop(HYBstart, HYBstop, HYBmili);
	if(Opt.verbose)
		cout << "verbose on. HYB GPU: " << HYBmili << endl;
	else
		cout << "HYB GPU: " << HYBmili << endl;

	for (int bin = 0; bin < 2 * Opt.nBin + 1; ++bin)
		cudaStreamDestroy(streams[bin]);
	// check correctness
	checkCuda(cudaMemcpy(&U[mode0].vals[0], dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE), cudaMemcpyDeviceToHost), 0);
	cudaFree(dVals); cudaFree(dCOOVals); cudaFree(dCSLVals); 
	cudaFree(dU0); cudaFree(dU1); cudaFree(dU2);
	cudaFree(dfbrIdx0); cudaFree(dInds2); cudaFree(dInds3); 
    cudaFree(dfbrIdx0); cudaFree(dfbrIdx1); cudaFree(dFbrIdx2);
    cudaFree(dfbrPtr0); cudaFree(dfbrPtr1); cudaFree(dFbrPtr2);
	cudaFree(dCSLInds1); cudaFree(dCSLInds2); cudaFree(dCSLSlcPtr); cudaFree(dCSLSlcInds);
	cudaFree(dCOOInds0); cudaFree(dCOOInds1); cudaFree(dCOOInds2); 

	return 0;
}