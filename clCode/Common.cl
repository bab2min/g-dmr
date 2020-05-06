
typedef uint VID;
typedef ushort TID;

TID sampleFromAccLikelihood(uint rand, const float* likelihoods)
{
	float nRand = (rand / 4294967296.f) * likelihoods[TMT_K - 1];
	/*
	#pragma unroll TMT_K
	for(TID k = 0; k < TMT_K; k++)
	{
		if (nRand < likelihoods[k]) return k;
	}
	/*/ 
	float4 nRand4 = (float4)nRand;
	TID k = 0;
	#pragma unroll TMT_K
	for (; k < (TMT_K >> 2) << 2; k += 4)
	{
		int4 r = nRand4 <= vload4(0, &likelihoods[k]);
		if (any(r)) return k + 4 + r.s0 + r.s1 + r.s2 + r.s3;
	}
	#pragma unroll TMT_K
	for(; k < TMT_K - 1; k++)
	{
		if (nRand < likelihoods[k]) return k;
	}
	//*/
	return TMT_K - 1;
}

void addWord(int count, VID v, TID z, uint* numByTopicDoc, global uint* numByWordTopic, local uint* numByTopic)
{
	//if (z >= TMT_K) return;
	numByTopicDoc[z] += count;

	if (count == 1)
	{
		atomic_inc(&numByWordTopic[TMT_K*v + z]);
		atomic_inc(&numByTopic[z]);
	}
	else if (count == -1)
	{
		atomic_dec(&numByWordTopic[TMT_K*v + z]);
		atomic_dec(&numByTopic[z]);
	}
}


float digamma(float x)
{
	return log(x + 4) - 0.5f / (x + 4) - 1 / 12.f / pow(x + 4, 2) + 1 / 120.f / pow(x + 4, 4) /*- 1 / 252.f / pow(x + 4, 6) + 1 / 240.f / pow(x + 4, 8)*/
		- 1 / (x + 3) - 1 / (x + 2) - 1 / (x + 1) - 1 / x;
}


void atomic_float_add_global(volatile global float *source, const float operand) {
	union {
		unsigned int intVal;
		float floatVal;
	} newVal;
	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;

	do {
		prevVal.floatVal = *source;
		newVal.floatVal = prevVal.floatVal + operand;
	} while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

void atomic_float_add_local(volatile local float *source, const float operand) {
	union {
		unsigned int intVal;
		float floatVal;
	} newVal;
	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;

	do {
		prevVal.floatVal = *source;
		newVal.floatVal = prevVal.floatVal + operand;
	} while (atomic_cmpxchg((volatile local unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

