#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <intrin.h>

//#include <tbb/parallel_for.h>
//#include <tbb/blocked_range.h>
//#include <tbb/atomic.h>
using namespace std;

struct pseudoSyn{
	int NStart;
	int NEnd;
	int Weight;
	int Delay;
};

void CountingSortNormal(vector<int> &VectorToBeSorted, 
	vector<int> &SortCompartmentVector, 
	vector<int> &OutVector, 
	vector<int> &EndIndexVect, 
	int Range){
	
	vector<int> StartIndexVect(Range, 0);
	int nElems = VectorToBeSorted.size();
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
	int nElems = VectorToBeSorted.size();
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
	vector<vector<int> > &WeightQueue,
	vector<vector<int> > &NEndQueue,
	vector<int> &Iin,
	vector<int> &VSim,
	vector<pseudoSyn> &SynVector, 
	vector<int> &preSynNeuronSectionBeg,
	vector<int> &preSynNeuronSectionEnd,
	vector<vector<int> > &NeuronSelectionVector,
	int nSteps,
	int nBins,
	int &TotalSpikes){
	
	int M = SynVector.size();
	int N = preSynNeuronSectionBeg.size();
	const int NCases = NeuronSelectionVector.size();

	WeightQueue.resize(nBins, vector<int>(0));
	NEndQueue.resize(nBins, vector<int>(0));
	VSim.resize(N, 0);
	vector<vector<int> > IndexQueue(nBins, vector<int>(0));
	vector<int> SpikeStoreWeightList;
	vector<int> SpikeStoreNEndList;
	vector<int> tempSpikeStoreIndexList;
	vector<int> SpikeStoreIndexList;
	vector<int> SpikeStoreDelayList;
	vector<int> EndIndexVect(nBins, 0);

	int CurrentQueue = 0;
	TotalSpikes = 0;
	int * StartPointerWeight;
	int * StartPointerNEnd;
	int * StartPointerIndex;
	int * EndPointerWeight;
	int * EndPointerNEnd;
	int * EndPointerIndex;
	int isdebug = false;
	for (int i = 0; i < nSteps; ++i){
		//if (i / 400 == 12400 / 400 || i % 400 == 0)
		//	cout << i << " steps complete." << endl;
		//if (i == 12424)
		//	isdebug = true;
		
		int temp = i % NCases;
		int CurrentQSize = WeightQueue[CurrentQueue].size();
		if (CurrentQSize){
			StartPointerWeight = &WeightQueue[CurrentQueue][0];
			StartPointerNEnd = &NEndQueue[CurrentQueue][0];
			StartPointerIndex = &IndexQueue[CurrentQueue][0];
			EndPointerWeight = &WeightQueue[CurrentQueue][CurrentQSize - 1] + 1;
			EndPointerNEnd = &NEndQueue[CurrentQueue][CurrentQSize - 1] + 1;
			EndPointerIndex = &IndexQueue[CurrentQueue][CurrentQSize - 1] + 1;
		}
		else{
			StartPointerWeight = StartPointerNEnd = 
			EndPointerWeight   = EndPointerNEnd   = 
			StartPointerIndex  = EndPointerIndex  = NULL;
		}

		for (int *iWeight = StartPointerWeight, *iNEnd = StartPointerNEnd, *iIndex = StartPointerIndex; 
			iWeight < EndPointerWeight; ++iWeight, ++iNEnd, ++iIndex){
			//BinVector[CurrentQueue][i] = BinVector[CurrentQueue][CurrentQSize - i - 1] + 2;
			Iin[*iNEnd - 1] += *iNEnd;
			Iin[*iNEnd - 1] -= *iIndex;
		}
		TotalSpikes += CurrentQSize;
		WeightQueue[CurrentQueue].clear();
		NEndQueue[CurrentQueue].clear();
		IndexQueue[CurrentQueue].clear();

		for (int j = 0; j < N; ++j){
			VSim[j] += (Iin[j] - VSim[j]);
			if (NeuronSelectionVector[temp][j]){
				int k = preSynNeuronSectionBeg[j];
				int kend = preSynNeuronSectionEnd[j];
				for (; k < kend; ++k){
					int temp = SynVector[k].Delay - 1;
					tempSpikeStoreIndexList.push_back(k);
					//SpikeStoreNEndList.push_back(SynVector[k].NEnd);
					//SpikeStoreWeightList.push_back(SynVector[k].Weight);
					SpikeStoreDelayList.push_back(temp);
				}
			}
		}
		CountingSortNormal(tempSpikeStoreIndexList, SpikeStoreDelayList,
			SpikeStoreIndexList, EndIndexVect, nBins);

		// creating arrays after having sorted
		for (int j = 0; j < nBins; ++j){
			int Start = (j == 0) ? 0 : EndIndexVect[j - 1];
			int End = EndIndexVect[j];
			int CurrentBin = (j + 1 + CurrentQueue) % nBins;
			for (int k = Start; k < End; ++k){
				WeightQueue[CurrentBin].push_back(SynVector[SpikeStoreIndexList[k]].Weight);
			}
		}
		for (int j = 0; j < nBins; ++j){
			int Start = (j == 0) ? 0 : EndIndexVect[j - 1];
			int End = EndIndexVect[j];
			int CurrentBin = (j + 1 + CurrentQueue) % nBins;
			for (int k = Start; k < End; ++k){
				NEndQueue[CurrentBin].push_back(SynVector[SpikeStoreIndexList[k]].NEnd);
			}
		}
		for (int j = 0; j < nBins; ++j){
			int Start = (j == 0) ? 0 : EndIndexVect[j - 1];
			int End = EndIndexVect[j];
			int CurrentBin = (j + 1 + CurrentQueue) % nBins;
			for (int k = Start; k < End; ++k){
				IndexQueue[CurrentBin].push_back(SpikeStoreIndexList[k]);
			}
		}
		
		tempSpikeStoreIndexList.clear();
		SpikeStoreDelayList.clear();
		for (int j = 0; j < nBins; ++j){
			EndIndexVect[j] = 0;
		}
		
		CurrentQueue = (CurrentQueue + 1) % nBins;
	}

	for (int i = 0; i < nBins; ++i){
		TotalSpikes += WeightQueue[i].size();
	}
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
		WeightQueue,
		NEndQueue,
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