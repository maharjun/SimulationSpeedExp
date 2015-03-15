#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <intrin.h>
#include "..\Headers\MexMem.hpp"
using namespace std;

struct pseudoSyn{
	int NStart;
	int NEnd;
	int Weight;
	int Delay;
};

inline void prefetch(const char *Pointer, int Size){
	size_t NoofPrefetches = ((size_t)Pointer + Size - 1) / (1 << 6) - ((size_t)Pointer) / (1 << 6) + 1;
	int offset = 0;
	for (int i = 0; i < NoofPrefetches; ++i){
		offset += 64;
		_mm_prefetch(Pointer + offset, 2);
	}
}

void CountingSortNormal(MexVector<pseudoSyn> &VectorToBeSorted, 
	MexVector<int> &SortCompartmentVector, 
	MexVector<pseudoSyn> &OutVector, 
	MexVector<int> &EndIndexVect, 
	int Range){
	
	for (int j = 0; j < Range; ++j){
		EndIndexVect[j] = 0;
	}
	MexVector<int> StartIndexVect(Range, 0);
	size_t nElems = VectorToBeSorted.size();
	for (int j = 0; j < nElems; ++j){
		EndIndexVect[SortCompartmentVector[j]]++;
	}
	for (int j = 1; j < Range; ++j){
		EndIndexVect[j] += EndIndexVect[j - 1];
		StartIndexVect[j] = EndIndexVect[j - 1];
	}
	if (OutVector.capacity() != VectorToBeSorted.capacity()) OutVector.reserve(VectorToBeSorted.capacity());
	if (OutVector.size() != nElems) OutVector.resize(nElems);
	for (int j = 0; j < nElems; ++j){
		OutVector[StartIndexVect[SortCompartmentVector[j]]] = VectorToBeSorted[j];
		StartIndexVect[SortCompartmentVector[j]]++;
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
		if (CacheVectIndices[CurrElem] == CacheSize){
			//prefetch(reinterpret_cast<const char*>(&OutVector + StartIndexVect[CurrElem] + CacheSize), CacheSize * sizeof(pseudoSyn));
			for (int k = StartIndexVect[CurrElem], l = CurrElem * CacheSize; k < StartIndexVect[CurrElem] + CacheSize; ++k, ++l){
				OutVector[k] = CacheSynVect[l];
			}
			CacheVectIndices[CurrElem] = 0;
			StartIndexVect[CurrElem] += CacheSize;
		}
		CacheSynVect[CurrElem * CacheSize + CacheVectIndices[CurrElem]] = VectorToBeSorted[j];
		++CacheVectIndices[CurrElem];
	}
	for (int i = 0; i < Range; ++i){
		for (int j = StartIndexVect[i], k = i * CacheSize; j < StartIndexVect[i] + CacheVectIndices[i]; ++j, ++k){
			OutVector[j] = CacheSynVect[k];
		}
	}

}

void inPlaceCountingSort(MexVector<int> &VectorToBeSorted, MexVector<int> &SortCompartmentVector, MexVector<int> &EndIndexVect, int Range){
	//_mm_prefetch((const char*)&VectorToBeSorted[0], 0);
	//_mm_prefetch((const char*)&SortCompartmentVector[0], 0);
	//EndIndexVect = MexVector<int>(Range, 0);    // Used for counting sort
	MexVector<int> StartIndexVect(Range, 0);

	// In Place Counting Sort
	size_t nElems = VectorToBeSorted.size();
	for (int j = 0; j < nElems; ++j){
		EndIndexVect[SortCompartmentVector[j]]++;
	}
	for (int j = 1; j < Range; ++j){
		EndIndexVect[j] += EndIndexVect[j - 1];
		StartIndexVect[j] = EndIndexVect[j - 1];
	}
	int CycleStart = Range;
	//int CurrentPosition;
	int CurrentDelayValue;
	int CurrentValue;
	for (int j = 0; j < Range; ++j)
		if (StartIndexVect[j] != EndIndexVect[j]){
			CycleStart = j;
			CurrentDelayValue = SortCompartmentVector[StartIndexVect[CycleStart]];
			CurrentValue = VectorToBeSorted[StartIndexVect[CycleStart]];
			break;
		}
	while (CycleStart < Range){
		int IndextoStoreIn = StartIndexVect[CurrentDelayValue];
		++(StartIndexVect[CurrentDelayValue]);
		if (CurrentDelayValue == CycleStart){
			VectorToBeSorted[IndextoStoreIn] = CurrentValue;
			for (; CycleStart < Range; ++CycleStart)
				if (StartIndexVect[CycleStart] != EndIndexVect[CycleStart]) break;
			if (CycleStart < Range){
				CurrentDelayValue = SortCompartmentVector[StartIndexVect[CycleStart]];
				CurrentValue = VectorToBeSorted[StartIndexVect[CycleStart]];
			}

		}
		else{
			int temp = CurrentValue;
			CurrentValue = VectorToBeSorted[IndextoStoreIn];
			CurrentDelayValue = SortCompartmentVector[IndextoStoreIn];
			VectorToBeSorted[IndextoStoreIn] = temp;
		}
	}
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
		pseudoSyn* tempPtr = &tempSpikeStoreList[0];
		for (size_t j = 0; j < NoofContigRegions; ++j){
			size_t SizeofContigRegion = CurrSpikingNeurons[4 * j + 1];
			//if (j < NoofContigRegions - 1){
			//	pseudoSyn *Ptrb4NextContigRegion = reinterpret_cast<pseudoSyn*>(CurrSpikingNeurons[4 * j + 3]);
			//	prefetch((const char*)Ptrb4NextContigRegion, SizeofContigRegion*sizeof(pseudoSyn));
			//}
			pseudoSyn *tempPtrSynVect = &SynVector[CurrSpikingNeurons[4 * j]];
			for (int k = 0; k < SizeofContigRegion; ++k){
				tempPtr[k] = tempPtrSynVect[k];
			}
			tempPtr += SizeofContigRegion;
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
	cout << "Number of actual iterations = " << TotalSpikesTemp;
}

int main(){
	mt19937 RandGen;
	int N = 10000;
	int onemsbyTstep = 4;
	int Range = onemsbyTstep*20;
	int nSteps = 8000 * onemsbyTstep;
	float ProbofSelecting = 1.0f / 50;
	float ProbofFiring = 1.0f / 1000;
	float LimitProbofFiring = (1.0f / 2) / onemsbyTstep;
	int TotalSpikes;
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

	SerialBinning(
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

	int TotalSpikesEstimate = 0;
	for (int i = 0; i < 8; ++i){
		for (int j = 0; j < N; ++j){
			if (NeuronSelectionVector[i][j]){
				TotalSpikesEstimate += NoofOutSyns[j];
			}
		}
	}
	cout << "Just 4 Kicks: " << pseudoV[1] << pseudoIin[1] << endl;
	TotalSpikesEstimate = (float)(TotalSpikesEstimate) / 8 * nSteps;
	if (WeightQueue.size() == Range){
		for (int i = 0; i < Range; ++i)
			cout << WeightQueue[i].capacity() << endl;
	}
	cout << "no of spikes estimated = " << TotalSpikesEstimate << endl;
	cout << "no of spikes stored = " << TotalSpikes << endl;
	cout << "Dis Shit Tuk " << chrono::duration_cast<chrono::milliseconds>(TEnd - TBeg).count() << " ms" << endl;
	system("pause");
}
