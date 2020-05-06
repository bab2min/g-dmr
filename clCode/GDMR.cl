
void calcAccLikelihoodGDMR(float* ret, VID v, const uint* numByTopicDoc, global const uint* numByWordTopic, local const uint* numByTopic, float* alpha, float beta)
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

float getTPFromTerms(constant float* lambdas, float* terms)
{ 
	float ret = 0;
#pragma unroll F
	for(uint f = 0; f < F; ++f)
	{
		ret += lambdas[TMT_K * f] * terms[f];
	}
	return ret;
}

kernel void trainGDMR(uint D, constant uint* WsOffsetByD, constant VID* Ws, constant uint* VDOffset, constant uint* VPartition,
	global TID* Zs, global uint* NByTopicDoc, global uint* NByWordTopic, global uint* NByTopic,
	constant float* lambdas, float beta, constant float* FByD,
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
	for (uint i = 0; i < iter; ++i)
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
				float alphaF[TMT_K];
				uint numByTopicDoc[TMT_K];
				float terms[F] = { 0, };
				getTerms(&FByD[TMT_F_STRIDE * d], terms);
				#pragma unroll TMT_K
				for (uint k = 0; k < TMT_K; ++k)
				{
					numByTopicDoc[k] = NByTopicDoc[TMT_K * d + k];
					alphaF[k] = exp(getTPFromTerms(&lambdas[k], terms)) + TMT_ALPHA_EPS;
				}
				for (uint w = vOffset[vId]; w < vOffset[vId + 1]; ++w)
				{
					VID v = WsD[w];
					addWord(-1, v, ZsD[w], numByTopicDoc, NByWordTopic, numByTopic);
					calcAccLikelihoodGDMR(likelihoods, v, numByTopicDoc, NByWordTopic, numByTopic, alphaF, beta);
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

kernel void calcObjGDMR(uint D, constant uint* WsOffsetByD,
	constant uint* NByTopicDoc, float sigma, constant float* FByD, constant float* x, global float* ret)
{
	uint groupId = get_group_id(0);
	uint groupSize = get_global_size(0) / get_local_size(0);

	uint gId = get_global_id(0);
	uint gSize = get_global_size(0);

	float tmp[F * TMT_K + 1] = { 0, };
	for (uint d = gId; d < D; d += gSize)
	{
		constant uint* numByTopicDoc = &NByTopicDoc[TMT_K * d];
		constant float* vx = &FByD[TMT_F_STRIDE * d];
		uint size = WsOffsetByD[d + 1] - WsOffsetByD[d];
		float ts[TMT_K] = { 0, };
		float alphaF[TMT_K];
		float alphaSum = 0;
		float terms[F] = { 0, };
		getTerms(vx, terms);
		for (int k = 0; k < TMT_K; ++k)
		{
			alphaF[k] = exp(getTPFromTerms(&x[k], terms)) + TMT_ALPHA_EPS;
			alphaSum += alphaF[k];
			tmp[0] += lgamma(alphaF[k]) - lgamma(numByTopicDoc[k] + alphaF[k]);
			ts[k] = -(digamma(alphaF[k]) - digamma(numByTopicDoc[k] + alphaF[k]));
		}
		tmp[0] -= lgamma(alphaSum) - lgamma(size + alphaSum);
		float t = digamma(alphaSum) - digamma(size + alphaSum);
		for (int k = 0; k < TMT_K; ++k)
		{
			float tsAlpha = (ts[k] + t) * alphaF[k];
#pragma unroll F
			for(int f = 0; f < F; ++f)
			{
				tmp[f * TMT_K + k + 1] -= tsAlpha * terms[f];
			}
		}
	}

	for(int i = 0; i <= F * TMT_K; ++i)
	{
		local float localSum;
		if (get_local_id(0) == 0) localSum = tmp[i];
		barrier(CLK_LOCAL_MEM_FENCE);
		if (get_local_id(0) > 0) atomic_float_add_local(&localSum, tmp[i]);
		barrier(CLK_LOCAL_MEM_FENCE);
		if (get_local_id(0) == 0) ret[(F * TMT_K + 1) * groupId + i] += localSum;
		barrier(CLK_LOCAL_MEM_FENCE);
	}}
