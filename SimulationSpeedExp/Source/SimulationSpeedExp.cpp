#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <emmintrin.h>

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/task_group.h>
#include <smmintrin.h>

#include "..\Headers\MexMem.hpp"
using namespace std;

struct pseudoSyn{
	int NStart;
	int NEnd;
	int Delay;
	int Weight;
};

struct pseudoSynRef{
	int  NStart;
	int  NEnd;
	int* WeightPtr;
};

inline void prefetch(const char *Pointer, int Size){
	size_t NoofPrefetches = ((size_t)Pointer + Size - 1) / (1 << 6) - ((size_t)Pointer) / (1 << 6) + 1;
	int offset = 0;
	for (int i = 0; i < NoofPrefetches; ++i){
		offset += 64;
		_mm_prefetch(Pointer + offset, 2);
	}
}

void CountingSortNormalNStart(MexVector<pseudoSynRef> &VectorToBeSorted,
	MexVector<pseudoSynRef> &OutVector,
	MexVector<int> &EndIndexVect,
	int Range){

	for (int j = 0; j < Range; ++j){
		EndIndexVect[j] = 0;
	}
	MexVector<int> StartIndexVect(Range, 0);
	size_t nElems = VectorToBeSorted.size();
	for (int j = 0; j < nElems; ++j){
		EndIndexVect[VectorToBeSorted[j].NStart-1]++;
	}
	for (int j = 1; j < Range; ++j){
		EndIndexVect[j] += EndIndexVect[j - 1];
		StartIndexVect[j] = EndIndexVect[j - 1];
	}
	if (OutVector.capacity() != VectorToBeSorted.capacity()) OutVector.reserve(VectorToBeSorted.capacity());
	if (OutVector.size() != nElems) OutVector.resize(nElems);
	for (int j = 0; j < nElems; ++j){
		int CurrKey = VectorToBeSorted[j].NStart - 1;
		OutVector[StartIndexVect[CurrKey]] = VectorToBeSorted[j];
		StartIndexVect[CurrKey]++;
	}
}
void CountingSortNormalNEnd(MexVector<pseudoSynRef> &VectorToBeSorted,
	MexVector<pseudoSynRef> &OutVector,
	MexVector<int> &EndIndexVect,
	int Range){

	for (int j = 0; j < Range; ++j){
		EndIndexVect[j] = 0;
	}
	MexVector<int> StartIndexVect(Range, 0);
	size_t nElems = VectorToBeSorted.size();
	for (int j = 0; j < nElems; ++j){
		EndIndexVect[VectorToBeSorted[j].NEnd - 1]++;
	}
	for (int j = 1; j < Range; ++j){
		EndIndexVect[j] += EndIndexVect[j - 1];
		StartIndexVect[j] = EndIndexVect[j - 1];
	}
	if (OutVector.capacity() != VectorToBeSorted.capacity()) OutVector.reserve(VectorToBeSorted.capacity());
	if (OutVector.size() != nElems) OutVector.resize(nElems);
	for (int j = 0; j < nElems; ++j){
		int CurrKey = VectorToBeSorted[j].NEnd - 1;
		OutVector[StartIndexVect[CurrKey]] = VectorToBeSorted[j];
		StartIndexVect[CurrKey]++;
	}
}
void CountingSortCached(MexVector<pseudoSyn> &VectorToBeSorted,
	//MexVector<int> &SortCompartmentVector,
	MexVector<pseudoSyn> &OutVector,
	MexVector<int> &EndIndexVect,
	int Range){
	const int CacheSize = 32;
	MexVector<pseudoSyn> CacheSynVect(Range * CacheSize);
	MexVector<size_t> CacheVectIndices(Range);

	for (int j = 0; j < Range; ++j){
		EndIndexVect[j] = 0;
	}
	MexVector<int> StartIndexVect(Range, 0);
	size_t nElems = VectorToBeSorted.size();
	for (int j = 0; j < nElems; ++j){
		EndIndexVect[VectorToBeSorted[j].Delay - 1]++;
	}
	for (int j = 1; j < Range; ++j){
		EndIndexVect[j] += EndIndexVect[j - 1];
		StartIndexVect[j] = EndIndexVect[j - 1];
	}
	if (OutVector.capacity() != VectorToBeSorted.capacity()) OutVector.reserve(VectorToBeSorted.capacity());
	if (OutVector.size() != nElems) OutVector.resize(nElems);
	for (int j = 0; j < nElems; ++j){
		int CurrElem = VectorToBeSorted[j].Delay - 1;

		size_t *CurrentIndexPtr = &CacheVectIndices[CurrElem];
		size_t CurrentIndex = *CurrentIndexPtr;
		if (CurrentIndex == CacheSize){
			//prefetch(reinterpret_cast<const char*>(&OutVector + StartIndexVect[CurrElem] + CacheSize), CacheSize * sizeof(pseudoSyn));
			__m128i* kbeg = reinterpret_cast<__m128i*>(&OutVector[StartIndexVect[CurrElem]]);
			__m128i* kend = reinterpret_cast<__m128i*>(&OutVector[StartIndexVect[CurrElem]]) + CacheSize;
			__m128i* lbeg = reinterpret_cast<__m128i*>(&CacheSynVect[CurrElem * CacheSize]);

			for (__m128i* k = kbeg, *l = lbeg; k < kend; k++, l++){
				_mm_stream_si128(k, _mm_load_si128(l)); 
			}
			CurrentIndex = 0;
			*CurrentIndexPtr = 0;
			StartIndexVect[CurrElem] += CacheSize;
		}
		_mm_store_si128( (__m128i*)CacheSynVect.begin() + CurrElem * CacheSize + CurrentIndex,
			_mm_load_si128((__m128i*)VectorToBeSorted.begin() + j));
		++(*CurrentIndexPtr);
	}
	for (int i = 0; i < Range; ++i){
		for (int j = StartIndexVect[i], k = i * CacheSize; j < StartIndexVect[i] + CacheVectIndices[i]; ++j, ++k){
			OutVector[j] = CacheSynVect[k];
		}
	}
	_mm_mfence();
}

void SerialBinning(
	MexVector<int> &Iin,
	MexVector<int> &VSim,
	MexVector<pseudoSyn> &SynVector, 
	MexVector<int> &preSynNeuronSectionBeg,
	MexVector<int> &preSynNeuronSectionEnd,
	MexVector<MexVector<int> > &NeuronSelectionVector,
	int nSteps,
	int nBins,
	int &TotalSpikes){
	
	size_t M = SynVector.size();
	size_t N = preSynNeuronSectionBeg.size();
	const size_t NCases = NeuronSelectionVector.size();

	VSim.resize(N, 0);
	MexVector<pseudoSyn> tempSpikeStoreList;
	MexVector<int> SpikeStoreDelayList;
	MexVector<int> StartEndIndexVect(nBins * 2, 0);
	MexVector<MexVector<pseudoSyn> > SpikeQueue(nBins, MexVector<pseudoSyn>(0));
	MexVector<MexVector<int> > EndIndices(nBins, MexVector<int>(nBins, 0));
	MexVector<size_t> CurrSpikingNeurons;
	CurrSpikingNeurons.reserve(N * 4);

	int CurrentQueue = 0;
	TotalSpikes = 0;
	size_t TotalSpikesTemp = 0;
	int isdebug = false;
	for (int i = 0; i < nSteps; ++i){
		//if (i / 400 == 12400 / 400 || i % 400 == 0)
		//	cout << i << " steps complete." << endl;
		//if (i == 12424)
		//	isdebug = true;
		
		int temp = i % NCases;
		//prefetch((const char*)&SpikeQueue[CurrentQueue][0], EndIndices[CurrentQueue][0]);

		for (int j = CurrentQueue; j < nBins + CurrentQueue; ++j){
			int ActualIndex = (j % nBins);
			StartEndIndexVect[2 * ActualIndex + 1] = EndIndices[ActualIndex][j - CurrentQueue];
		}
		StartEndIndexVect[2 * CurrentQueue] = 0;
		for (int j = CurrentQueue; j < nBins + CurrentQueue; ++j){
			int ActualIndex = (j % nBins);
			
			const MexVector<pseudoSyn>::iterator SynStartPointer = SpikeQueue[ActualIndex].begin() + StartEndIndexVect[2 * ActualIndex];
			const MexVector<pseudoSyn>::iterator SynEndPointer = SpikeQueue[ActualIndex].begin() + StartEndIndexVect[2 * ActualIndex + 1];
			TotalSpikesTemp += (SynEndPointer - SynStartPointer);
			for (MexVector<pseudoSyn>::iterator iSyn = SynStartPointer; iSyn < SynEndPointer; ++iSyn){
				Iin[iSyn->NEnd - 1] += iSyn->Weight;
				Iin[iSyn->NEnd - 1] -= iSyn->Delay;
			}
			StartEndIndexVect[2 * ActualIndex] = StartEndIndexVect[2 * ActualIndex + 1];
		}
		int NextQueue = (CurrentQueue + nBins - 1) % nBins;
		SpikeQueue[NextQueue].clear();
		int CacheBuffering = 16;	// Each time a cache of size 64 will be pulled in 
		size_t TotalNoofCurrentSpikes = 0;
		for (int j = 0, CurrentContig = -1, PrevSpikedNeuron = -1; j < N; ++j){
			VSim[j] += (Iin[j] - VSim[j]);
			if (NeuronSelectionVector[temp][j]){
				int k = preSynNeuronSectionBeg[j];
				int kend = preSynNeuronSectionEnd[j];
				if (k != kend){
					TotalNoofCurrentSpikes += kend - k;
					if (CurrentContig == -1 || 
						preSynNeuronSectionEnd[PrevSpikedNeuron] != preSynNeuronSectionBeg[j]){
						CurrentContig++;
						CurrSpikingNeurons.push_back(k);
						CurrSpikingNeurons.push_back(kend - k);
						CurrSpikingNeurons.push_back(TotalNoofCurrentSpikes - kend + k);
						CurrSpikingNeurons.push_back(reinterpret_cast<size_t>(&SynVector[0] + k));
					}
					else{
						CurrSpikingNeurons[4 * CurrentContig + 1] += kend - k;
					}
					PrevSpikedNeuron = j;
				}
			}
		}
		size_t NoofContigRegions = CurrSpikingNeurons.size() / 4;
		if (TotalNoofCurrentSpikes > tempSpikeStoreList.capacity()){
			int currCapacity = tempSpikeStoreList.capacity();
			while (TotalNoofCurrentSpikes > currCapacity){
				if (currCapacity == 0)
					currCapacity = 4;
				else
					currCapacity = currCapacity + (currCapacity >> 1);
			}
			tempSpikeStoreList.reserve(currCapacity);
		}
			
		tempSpikeStoreList.resize(TotalNoofCurrentSpikes);
		__m128i* tempPtr = reinterpret_cast<__m128i*>(&tempSpikeStoreList[0]);
		for (size_t j = 0; j < NoofContigRegions; ++j){
			size_t SizeofContigRegion = CurrSpikingNeurons[4 * j + 1];
			//if (j < NoofContigRegions - 1){
			//	pseudoSyn *Ptrb4NextContigRegion = reinterpret_cast<pseudoSyn*>(CurrSpikingNeurons[4 * j + 3]);
			//	prefetch((const char*)Ptrb4NextContigRegion, SizeofContigRegion*sizeof(pseudoSyn));
			//}
			__m128i *ContigRegionBeg = reinterpret_cast<__m128i*>(&SynVector[CurrSpikingNeurons[4 * j]]);
			__m128i *ContigRegionEnd = reinterpret_cast<__m128i*>(&SynVector[CurrSpikingNeurons[4 * j]])
			                                                      + SizeofContigRegion;
			
			// 64 aligned pointers
			__m128i *ContigRegionBeg64 = reinterpret_cast<__m128i*>
											(((reinterpret_cast<size_t>(ContigRegionBeg) >> 6) + 1) << 6);
			__m128i *ContigRegionEnd64 = reinterpret_cast<__m128i*>
											((reinterpret_cast<size_t>(ContigRegionEnd) >> 6) << 6);

			// Streaming 64 aligned
			for (__m128i *ContigRegionPtr = ContigRegionBeg; 
				ContigRegionPtr < ContigRegionEnd; 
				++ContigRegionPtr, ++tempPtr){
				__m128i xmm0 = _mm_load_si128(ContigRegionPtr);
				_mm_stream_si128(tempPtr, xmm0);
			}

		}
		CountingSortCached(tempSpikeStoreList, /*SpikeStoreDelayList,*/
			SpikeQueue[NextQueue], EndIndices[NextQueue], nBins);
		
		size_t CurrentQSize = tempSpikeStoreList.size();
		CurrSpikingNeurons.clear();
		tempSpikeStoreList.clear();
		//SpikeStoreDelayList.clear();
		TotalSpikes += CurrentQSize;
		CurrentQueue = NextQueue;
	}
	std::cout << "Number of actual iterations = " << TotalSpikesTemp;
}

void CachedBinning(
	MexVector<int> &Iin,
	MexVector<int> &VSim,
	MexVector<pseudoSyn> &SynVector,
	MexVector<int> &preSynNeuronSectionBeg,
	MexVector<int> &preSynNeuronSectionEnd,
	MexVector<MexVector<int> > &NeuronSelectionVector,
	int nSteps,
	int nBins,
	size_t &TotalSpikes
	){

	size_t M = SynVector.size();
	size_t N = preSynNeuronSectionBeg.size();
	const size_t NCases = NeuronSelectionVector.size();

	VSim.resize(N, 0);
	MexVector<MexVector<pseudoSynRef> > SpikeQueue(nBins, MexVector<pseudoSynRef>(0));

	int CurrentQueue = 0;
	TotalSpikes = 0;
	//size_t TotalSpikesTemp = 0;
	int isdebug = false;
	MexVector<pseudoSynRef> NEndSortedSpikes, NStartNEndSortedSpikes;
	MexVector<int> EndIndexVect(N, 0);
	chrono::time_point<chrono::system_clock> CountingSortTimeBeg, CountingSortTimeEnd;
	chrono::time_point<chrono::system_clock> CurrentProcingTimeBeg, CurrentProcingTimeEnd;
	chrono::time_point<chrono::system_clock> SpikeStoreTimeBeg, SpikeStoreTimeEnd;
	int CountingSortDuration = 0;
	int CurrentProcingDuration = 0;
	int SpikeStoreDuration = 0;

	for (int i = 0; i < nSteps; ++i){
		int temp = i % NCases;

		int CurrentQSize = SpikeQueue[CurrentQueue].size();
		MexVector<pseudoSynRef>::iterator StartPointer, EndPointer;
		// performing a normal countingsort without fancy caching
		CountingSortTimeBeg = chrono::system_clock::now();

		CountingSortNormalNEnd(SpikeQueue[CurrentQueue], NEndSortedSpikes, EndIndexVect, N);
		CountingSortNormalNStart(NEndSortedSpikes, NStartNEndSortedSpikes, EndIndexVect, N);

		CountingSortTimeEnd = chrono::system_clock::now();
		CountingSortDuration += chrono::duration_cast<chrono::microseconds>(CountingSortTimeEnd - CountingSortTimeBeg).count();

		if (CurrentQSize){
			StartPointer = NStartNEndSortedSpikes.begin();
			EndPointer = NStartNEndSortedSpikes.end();
		}
		else{
			StartPointer = EndPointer = NULL;
		}
		CurrentProcingTimeBeg = chrono::system_clock::now();
		for (MexVector<pseudoSynRef>::iterator iSpike = StartPointer;
			iSpike < EndPointer; ++iSpike){
			int temp = *(iSpike->WeightPtr);
			Iin[iSpike->NEnd - 1] += temp;
			*(iSpike->WeightPtr) = -temp + Iin[iSpike->NEnd - 1];
		}
		CurrentProcingTimeEnd = chrono::system_clock::now();
		CurrentProcingDuration += chrono::duration_cast<chrono::microseconds>(CurrentProcingTimeEnd - CurrentProcingTimeBeg).count();

		TotalSpikes += CurrentQSize;
		SpikeQueue[CurrentQueue].clear();
		int NextQueue = (CurrentQueue + 1) % nBins;

		int CacheBuffering = 64;	// Each time a cache of size 64 will be pulled in 
		MexVector<__m128i> BinningBuffer(CacheBuffering*nBins);	//each element is 16 bytes
		MexVector<int> BufferInsertIndex(nBins, 0);

		SpikeStoreTimeBeg = chrono::system_clock::now();
		for (int j = 0; j < N; ++j){
			VSim[j] += (Iin[j] - VSim[j]);
			if (NeuronSelectionVector[temp][j]){
				int k = preSynNeuronSectionBeg[j];
				int kend = preSynNeuronSectionEnd[j];
				if (k != kend){
					int NoofCurrNeuronSpikes = kend - k;
					MexVector<pseudoSyn>::iterator iSyn = SynVector.begin() + k;

					MexVector<pseudoSyn>::iterator iSynEnd = SynVector.begin() + kend;
					//TotalSpikesTemp += iSynEnd - iSyn;
					for (; iSyn < iSynEnd; ++iSyn){
						int CurrIndex = (CurrentQueue + iSyn->Delay) % nBins;
						int BufferIndex = BufferInsertIndex[CurrIndex];
						int *BufferIndexPtr = &BufferInsertIndex[CurrIndex];
						if (BufferIndex == CacheBuffering){
							SpikeQueue[CurrIndex].push_size(CacheBuffering);
							//TotalSpikesTemp += CacheBuffering;
							__m128i* kbeg = reinterpret_cast<__m128i*>(SpikeQueue[CurrIndex].end() - CacheBuffering);
							__m128i* kend = reinterpret_cast<__m128i*>(SpikeQueue[CurrIndex].end());
							__m128i* lbeg = reinterpret_cast<__m128i*>(BinningBuffer.begin() + CurrIndex * CacheBuffering);

							for (__m128i* k = kbeg, *l = lbeg; k < kend; k++, l++){
								_mm_stream_si128(k, _mm_load_si128(l));
							}
							BufferIndex = 0;
							*BufferIndexPtr = 0;
						}
						__m128i xmm1 = _mm_load_si128((__m128i*)iSyn);
						__m128i xmm2; xmm2.m128i_i64[0] = reinterpret_cast<size_t>(iSyn)+offsetof(pseudoSyn, Weight);
						
						BinningBuffer[CurrIndex*CacheBuffering + BufferIndex] = _mm_unpacklo_epi64(xmm1, xmm2);
						++*BufferIndexPtr;
					}
				}
			}
		}
		for (int i = 0; i < nBins; ++i){
			size_t CurrNElems = BufferInsertIndex[i];
			SpikeQueue[i].push_size(CurrNElems);
			__m128i* kbeg = reinterpret_cast<__m128i*>(SpikeQueue[i].end() - CurrNElems);
			__m128i* kend = reinterpret_cast<__m128i*>(SpikeQueue[i].end());
			__m128i* lbeg = reinterpret_cast<__m128i*>(BinningBuffer.begin() + i * CacheBuffering);
			for (__m128i* k = kbeg, *l = lbeg; k < kend; ++k, ++l){
				_mm_stream_si128(k, _mm_load_si128(l));
			}
			BufferInsertIndex[i] = 0;
		}
		SpikeStoreTimeEnd = chrono::system_clock::now();
		SpikeStoreDuration += chrono::duration_cast<chrono::microseconds>(SpikeStoreTimeEnd - SpikeStoreTimeBeg).count();
		CurrentQueue = NextQueue;
	}
	for (int i = 0; i < nBins; ++i){
		int CurrentQueueTemp = (CurrentQueue + i) % nBins;
		TotalSpikes += SpikeQueue[CurrentQueueTemp].size();
	}
	
	std::cout << "Time Spent in Counting Sort = " << CountingSortDuration / 1000 << " milliseconds" << endl;
	std::cout << "Time Spent in Current Processing = " << CurrentProcingDuration / 1000 << " milliseconds" << endl;
	std::cout << "Time Spent in Spike Storing = " << SpikeStoreDuration / 1000 << " milliseconds" << endl;

}
int main(){
	mt19937 RandGen;
	int N = 10000;
	int onemsbyTstep = 4;
	int Range = onemsbyTstep*20;
	int nSteps = 8000 * onemsbyTstep;
	float ProbofSelecting = 1.0f / 50;
	float ProbofFiring = 1.0f / 600;
	float LimitProbofFiring = (2.0f / 3) / onemsbyTstep;
	size_t TotalSpikes;
	RandGen.seed(28);
	uniform_int_distribution<> RandDist(1, Range);
	uniform_real_distribution<double> RandFloatDist(0.0f, 1.0f);
	bernoulli_distribution BerDist(ProbofSelecting);

	MexVector<MexVector<int> > NeuronSelectionVector(8, MexVector<int>(N,0));
	MexVector<int> pseudoIin(N, 0);
	MexVector<int> pseudoV(N, 0);
	MexVector<int> preSynNeuronSectionBeg(N, -1);
	MexVector<int> preSynNeuronSectionEnd(N, -1);
	MexVector<int> NoofOutSyns(N, 0);
	MexVector<int> NoofInSyns(N, 0);
	MexVector<MexVector<int> > WeightQueue, NEndQueue;
	MexVector<pseudoSyn> Synapses;

	// filling preSynNeuronSectionBeg, preSynNeuronSectionBeg, and creating network.
	int TotalNoofSyns = 0;
	for (int i = 0; i < N; ++i){
		for (int j = i + 1; j < N; ++j){
			bool temp = BerDist(RandGen);
			if (temp){
				Synapses.push_back(pseudoSyn{ i + 1, j + 1, 0, 0 });
			}
			NoofOutSyns[i] += temp;
			NoofInSyns[j] += temp;
		}
		if (NoofOutSyns[i]){
			preSynNeuronSectionBeg[i] = TotalNoofSyns;
			preSynNeuronSectionEnd[i] = TotalNoofSyns + NoofOutSyns[i];
		}
		TotalNoofSyns += NoofOutSyns[i];
	}
	
	// Creating Synapse delays
	for (int i = 0; i < TotalNoofSyns; ++i){
		Synapses[i].Delay = RandDist(RandGen); // 1-20
	}
	// Creating Synapse Weights
	for (int i = 0; i < TotalNoofSyns; ++i){
		Synapses[i].Weight = (RandDist(RandGen) - 1) / 4 + 1; // 1-5
	}

	// filling up neuron selections
	MexVector<float> tempRandVect(N);
	 
	for (int i = 0; i < 8; ++i){
		for (int j = 0; j < N; ++j){
			NeuronSelectionVector[i][j] = (int)(RandFloatDist(RandGen) +
				((NoofInSyns[j] * ProbofFiring > LimitProbofFiring) ? LimitProbofFiring : NoofInSyns[j] * ProbofFiring));
		}
	}
	
	auto TBeg = chrono::system_clock::now();

	CachedBinning(
		pseudoIin,
		pseudoV,
		Synapses,
		preSynNeuronSectionBeg,
		preSynNeuronSectionEnd,
		NeuronSelectionVector,
		nSteps,
		Range,
		TotalSpikes
		);
	auto TEnd = chrono::system_clock::now();

	size_t TotalSpikesEstimate = 0;
	for (int i = 0; i < 8; ++i){
		for (int j = 0; j < N; ++j){
			if (NeuronSelectionVector[i][j]){
				TotalSpikesEstimate += NoofOutSyns[j];
			}
		}
	}
	std::cout << "Just 4 Kicks: " << pseudoV[1] << pseudoIin[1] << endl;
	TotalSpikesEstimate = (float)(TotalSpikesEstimate) / 8 * nSteps;
	if (WeightQueue.size() == Range){
		for (int i = 0; i < Range; ++i)
			std::cout << WeightQueue[i].capacity() << endl;
	}
	std::cout << "no of spikes estimated = " << TotalSpikesEstimate << endl;
	std::cout << "no of spikes stored = " << TotalSpikes << endl;
	std::cout << "Dis Shit Tuk " << chrono::duration_cast<chrono::milliseconds>(TEnd - TBeg).count() << " ms" << endl;
	std::system("pause");
}
