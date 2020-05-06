
// TMT_K : number of topics
// TMT_V : number of vocabs
// NRT : size of random table
// SIZE : size of local work


void calcAccLikelihood(float* ret, VID v, const uint* numByTopicDoc, global const uint* numByWordTopic, local const uint* numByTopic, constant float* alpha, float beta)
{
	//if (v >= TMT_V) return;
	float sum = 0;
	#pragma unroll TMT_K
	for (uint k = 0; k < TMT_K; ++k)
	{
		sum += (numByTopicDoc[k] + alpha[k]) * (numByWordTopic[TMT_K*v + k] + beta) / (numByTopic[k] + TMT_V * beta);
		ret[k] = sum;
	}
}

kernel void trainLDA(uint D, constant uint* WsOffsetByD, constant VID* Ws, constant uint* VDOffset, constant uint* VPartition,
		global TID* Zs, global uint* NByTopicDoc, global uint* NByWordTopic, global uint* NByTopic,
		constant float* alpha, float beta,
		uint iter, uint randSeed)
{
	float likelihoods[TMT_K];
	local uint numByTopic[TMT_K];
	local uint oldNumByTopic[TMT_K];

	uint groupId = get_group_id(0);
	uint groupSize = get_global_size(0) / get_local_size(0);

	uint gId = get_global_id(0);
	uint gSize = get_global_size(0);

	mwc64x_state_t rng;
	MWC64X_SeedStreams(&rng, randSeed % 4096, 4096);
	for(uint i = 0; i < iter; ++i)
	{
		for (uint p = 0; p < groupSize; ++p)
		{
			uint vId = (p + groupId + randSeed % groupSize) % groupSize;
			#pragma unroll TMT_K
			for (uint k = get_local_id(0); k < TMT_K; k += get_local_size(0)) oldNumByTopic[k] = numByTopic[k] = NByTopic[k];
			barrier(CLK_LOCAL_MEM_FENCE);
			for (uint d = gId; d < D; d += gSize)
			{
				constant VID* WsD = &Ws[WsOffsetByD[d]];
				global TID* ZsD = &Zs[WsOffsetByD[d]];
				constant uint* vOffset = &VDOffset[d * (groupSize + 1)];
				uint numByTopicDoc[TMT_K];
				#pragma unroll TMT_K
				for (uint k = 0; k < TMT_K; ++k) numByTopicDoc[k] = NByTopicDoc[TMT_K * d + k];
				for (uint w = vOffset[vId]; w < vOffset[vId + 1]; ++w)
				{
					VID v = WsD[w];
					addWord(-1, v, ZsD[w], numByTopicDoc, NByWordTopic, numByTopic);
					calcAccLikelihood(likelihoods, v, numByTopicDoc, NByWordTopic, numByTopic, alpha, beta);
					ZsD[w] = sampleFromAccLikelihood(MWC64X_NextUint(&rng), likelihoods);
					addWord(1, v, ZsD[w], numByTopicDoc, NByWordTopic, numByTopic);
				}
				#pragma unroll TMT_K
				for (uint k = 0; k < TMT_K; ++k) NByTopicDoc[TMT_K * d + k] = numByTopicDoc[k];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll TMT_K
			for (uint k = get_local_id(0); k < TMT_K; k += get_local_size(0)) atom_add(&NByTopic[k], numByTopic[k] - oldNumByTopic[k]);
			barrier(CLK_GLOBAL_MEM_FENCE);
		}
	}
}
