#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <intrin.h>

using namespace std;

struct pseudoSyn{
	int NStart;
	int NEnd;
	int Weight;
	int Delay;
};

void CountingSortNormal(vector<pseudoSyn> &VectorToBeSorted, 
	vector<int> &SortCompartmentVector, 
	vector<pseudoSyn> &OutVector, 
	vector<int> &EndIndexVect, 
	int Range){
	
	for (int j = 0; j < Range; ++j){
		EndIndexVect[j] = 0;
	}
	vector<int> StartIndexVect(Range, 0);
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

void inPlaceCountingSort(vector<int> &VectorToBeSorted, vector<int> &SortCompartmentVector, vector<int> &EndIndexVect, int Range){
	//_mm_prefetch((const char*)&VectorToBeSorted[0], 0);
	//_mm_prefetch((const char*)&SortCompartmentVector[0], 0);
	//EndIndexVect = vector<int>(Range, 0);    // Used for counting sort
	vector<int> StartIndexVect(Range, 0);

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

inline void prefetch(const char *Pointer, int Size){
	size_t NoofPrefetches = ((size_t)Pointer + Size - 1) / (1 << 6) - ((size_t)Pointer) / (1 << 6) + 1;
	int offset = 0;
	for (int i = 0; i < NoofPrefetches; ++i){
		offset += 64;
		_mm_prefetch(Pointer + offset, 0);
	}
}

void SerialBinning(
	vector<int> &Iin,
	vector<int> &VSim,
	vector<pseudoSyn> &SynVector, 
	vector<int> &preSynNeuronSectionBeg,
	vector<int> &preSynNeuronSectionEnd,
	vector<vector<int> > &NeuronSelectionVector,
	int nSteps,
	int nBins,
	int &TotalSpikes){
	
	size_t M = SynVector.size();
	size_t N = preSynNeuronSectionBeg.size();
	const size_t NCases = NeuronSelectionVector.size();

	VSim.resize(N, 0);
	vector<pseudoSyn> tempSpikeStoreList;
	vector<int> SpikeStoreDelayList;
	vector<int> StartEndIndexVect(nBins * 2, 0);
	vector<vector<pseudoSyn> > SpikeQueue(nBins, vector<pseudoSyn>(0));
	vector<vector<int> > EndIndices(nBins, vector<int>(nBins, 0));

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
			//prefetch(SpikeQueue[(j+1)%nBins])
			int ActualIndex = (j % nBins);
			
			const vector<pseudoSyn>::iterator SynStartPointer = SpikeQueue[ActualIndex].begin() + StartEndIndexVect[2 * ActualIndex];
			const vector<pseudoSyn>::iterator SynEndPointer = SpikeQueue[ActualIndex].begin() + StartEndIndexVect[2 * ActualIndex + 1];
			TotalSpikesTemp += (SynEndPointer - SynStartPointer);
			for (vector<pseudoSyn>::iterator iSyn = SynStartPointer; iSyn < SynEndPointer; ++iSyn){
				Iin[iSyn->NEnd - 1] += iSyn->Weight;
				Iin[iSyn->NEnd - 1] -= iSyn->Delay;
			}
			StartEndIndexVect[2 * ActualIndex] = StartEndIndexVect[2 * ActualIndex + 1];
		}
		int NextQueue = (CurrentQueue + nBins - 1) % nBins;
		SpikeQueue[NextQueue].clear();
		
		for (int j = 0; j < N; ++j){
			VSim[j] += (Iin[j] - VSim[j]);
			if (NeuronSelectionVector[temp][j]){
				int k = preSynNeuronSectionBeg[j];
				int kend = preSynNeuronSectionEnd[j];
				for (; k < kend; ++k){
					int temp = SynVector[k].Delay - 1;
					tempSpikeStoreList.push_back(SynVector[k]);
					SpikeStoreDelayList.push_back(temp);
				}
			}
		}
		CountingSortNormal(tempSpikeStoreList, SpikeStoreDelayList,
			SpikeQueue[NextQueue], EndIndices[NextQueue], nBins);
		
		size_t CurrentQSize = tempSpikeStoreList.size();
		tempSpikeStoreList.clear();
		SpikeStoreDelayList.clear();
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
	float ProbofFiring = 1.0f / 2000;
	float LimitProbofFiring = (1.0f / 2) / onemsbyTstep;
	int TotalSpikes;
	RandGen.seed(28);
	uniform_int_distribution<> RandDist(1, Range);
	uniform_real_distribution<double> RandFloatDist(0.0f, 1.0f);
	bernoulli_distribution BerDist(ProbofSelecting);

	vector<vector<int> > NeuronSelectionVector(8, vector<int>(N,0));
	vector<int> pseudoIin(N, 0);
	vector<int> pseudoV(N, 0);
	vector<int> preSynNeuronSectionBeg(N, -1);
	vector<int> preSynNeuronSectionEnd(N, -1);
	vector<int> NoofOutSyns(N, 0);
	vector<int> NoofInSyns(N, 0);
	vector<vector<int> > WeightQueue, NEndQueue;
	vector<pseudoSyn> Synapses;

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
	vector<float> tempRandVect(N);
	 
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