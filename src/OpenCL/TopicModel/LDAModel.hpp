#pragma once
#include "../OpenCLMgr.h"
#include "../../TopicModel/LDAModel.hpp"
#include "../../Utils/Utils.hpp"

/*
Implementation of LDA using Parallel Collapsed Gibbs Sampling on GPU by bab2min

* Yan, F., Xu, N., & Qi, Y. (2009). Parallel inference for latent dirichlet allocation on graphics processing units. In Advances in Neural Information Processing Systems (pp. 2134-2142).
*/

namespace tomoto
{
	namespace ocl
	{
		class ICL
		{
		public:
			virtual std::string selectDevice(size_t id) = 0;
			virtual void setWorkSize(size_t numGroup, size_t localThreadSize) = 0;
			virtual void setUpdate(size_t _clUpdate) = 0;
		};

		template<TermWeight _TW, 
			typename _Interface = ILDAModel,
			typename _Derived = void,
			typename _DocType = DocumentLDA<_TW, flags::continuous_doc_data>,
			typename _ModelState = ModelStateLDA<_TW>,
			template<TermWeight, size_t,
			typename ...> class _BaseModel = LDAModel
		>
		class CL_LDAModel : public _BaseModel<_TW, flags::continuous_doc_data, _Interface, 
			typename std::conditional<std::is_same<_Derived, void>::value, CL_LDAModel<_TW>, _Derived>::type,
			_DocType, _ModelState>, public ICL
		{
		protected:
			using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, CL_LDAModel<_TW>, _Derived>::type;
			using BaseClass = _BaseModel<_TW, flags::continuous_doc_data, _Interface, DerivedClass, _DocType, _ModelState>;
			using WeightType = typename BaseClass::WeightType;
			friend BaseClass;
			friend typename BaseClass::BaseClass;

			size_t numGroup = 32;
			size_t numLocalThread = 128;
			size_t clUpdate = 1;

			mutable OpenCLMgr clMgr;

			cl::Buffer clBufWs, clBufWsOffset, clBufZs, clBufVPartition, clBufVDOffset,
				clBufNumByTopicDoc, clBufNumByWordTopic, clBufNumByTopic, clBufRandTable, clBufAlphas;
			std::vector<uint32_t> vChunkOffset;
			std::vector<uint32_t> chunkOffsetByDoc;

			void initializeBuffer()
			{
				const size_t V = this->realV;

				for (size_t i = 0; i <= numGroup; ++i) vChunkOffset.emplace_back(V * i / numGroup);
				for (auto& doc : this->docs)
				{
					chunkOffsetByDoc.emplace_back(0);
					for (size_t i = 1; i <= numGroup; ++i)
					{
						auto vNext = vChunkOffset[i];
						chunkOffsetByDoc.emplace_back(
							std::find_if(doc.words.begin(), doc.words.end(), [vNext](VID v) { return v >= vNext; })
							- doc.words.begin());
					}
				}

				if (this->docs.empty()) return;

				clBufWs = cl::Buffer(clMgr.getContext(), CL_MEM_READ_ONLY, sizeof(VID) * this->words.size());
				clBufWsOffset = cl::Buffer(clMgr.getContext(), CL_MEM_READ_ONLY, sizeof(uint32_t) * this->wOffsetByDoc.size());
				clBufVPartition = cl::Buffer(clMgr.getContext(), CL_MEM_READ_ONLY, sizeof(uint32_t) * (numGroup + 1));
				clBufVDOffset = cl::Buffer(clMgr.getContext(), CL_MEM_READ_ONLY, sizeof(uint32_t) * this->docs.size() * (numGroup + 1));
				clBufZs = cl::Buffer(clMgr.getContext(), CL_MEM_READ_WRITE, sizeof(TID) * this->words.size());
				clBufNumByTopicDoc = cl::Buffer(clMgr.getContext(), CL_MEM_READ_WRITE, sizeof(uint32_t) * this->K * this->docs.size());
				clBufNumByWordTopic = cl::Buffer(clMgr.getContext(), CL_MEM_READ_WRITE, sizeof(uint32_t) * this->K * V);
				clBufNumByTopic = cl::Buffer(clMgr.getContext(), CL_MEM_READ_WRITE, sizeof(uint32_t) * this->K);
				clBufAlphas = cl::Buffer(clMgr.getContext(), CL_MEM_READ_ONLY, sizeof(float) * this->K);

				auto& queue = clMgr.getQueue();
				queue.enqueueWriteBuffer(clBufWs, true, 0, sizeof(VID) * this->words.size(), &this->words[0]);
				queue.enqueueWriteBuffer(clBufWsOffset, true, 0, sizeof(uint32_t) * this->wOffsetByDoc.size(), &this->wOffsetByDoc[0]);
				queue.enqueueWriteBuffer(clBufVPartition, true, 0, sizeof(uint32_t) * (numGroup + 1), &vChunkOffset[0]);
				queue.enqueueWriteBuffer(clBufZs, true, 0, sizeof(TID) * this->sharedZs.size(), &this->sharedZs[0]);
				queue.enqueueWriteBuffer(clBufNumByTopicDoc, true, 0, sizeof(uint32_t) * this->numByTopicDoc.size(), this->numByTopicDoc.data());
				queue.enqueueWriteBuffer(clBufVDOffset, true, 0, sizeof(uint32_t) * chunkOffsetByDoc.size(), &chunkOffsetByDoc[0]);
				queue.enqueueWriteBuffer(clBufNumByWordTopic, true, 0, sizeof(uint32_t) * this->K * V, this->globalState.numByTopicWord.data());
				queue.enqueueWriteBuffer(clBufNumByTopic, true, 0, sizeof(uint32_t) * this->K, &this->globalState.numByTopic[0]);
				queue.enqueueWriteBuffer(clBufAlphas, true, 0, sizeof(float) * this->K, this->alphas.data());
			}

			void prepareDoc(_DocType& doc, WeightType* topicDocPtr, size_t wordSize) const
			{
				sortAndWriteOrder(doc.words, doc.wOrder);
				BaseClass::prepareDoc(doc, topicDocPtr, wordSize);
			}

			void prepareCL()
			{
				const size_t V = this->realV;
				std::string err;
				if (clMgr.buildProgramFromFile({ "clCode/Common.cl", "clCode/mwc64x.cl", "clCode/LDA.cl" }, "", text::format("-D TMT_K=%d -D TMT_V=%d ", this->K, V), &err))
				{
					std::cerr << err << std::endl;
					throw cl::Error(-1);
				}
				initializeBuffer();
			}

			void receiveSamplingResult(cl::CommandQueue& queue)
			{
				const size_t V = this->realV;
				const auto K = this->K;
				try
				{
					queue.enqueueReadBuffer(clBufZs, true, 0, sizeof(TID) * this->sharedZs.size(), &this->sharedZs[0]);
					queue.enqueueReadBuffer(clBufNumByTopicDoc, true, 0, sizeof(uint32_t) * this->numByTopicDoc.size(), this->numByTopicDoc.data());
					queue.enqueueReadBuffer(clBufNumByWordTopic, true, 0, sizeof(uint32_t) * K * V, this->globalState.numByTopicWord.data());
					queue.enqueueReadBuffer(clBufNumByTopic, true, 0, sizeof(uint32_t) * K, this->globalState.numByTopic.data());
				}
				catch (const cl::Error& e)
				{
					std::cerr << OpenCLMgr::translateOpenCLError(e.err()) << " in " << e.what();
					throw;
				}
				//printf("Copy From GPU: %g ms\n", timer.getAndReset());
			}

			void optimizeParameters(ThreadPool& pool, _ModelState* localData, RandGen* rgs)
			{
				BaseClass::optimizeParameters(pool, localData, rgs);
				auto& queue = clMgr.getQueue();
				queue.enqueueWriteBuffer(clBufAlphas, true, 0, sizeof(float) * this->K, this->alphas.data());
			}

		public:
			using BaseClass::BaseClass;

			std::string selectDevice(size_t id) override
			{
				auto* di = clMgr.createDevice(id);
				if (!di) return "";
				return di->name;
			}

			void setWorkSize(size_t numGroup, size_t localThreadSize) override
			{ 
				this->numGroup = numGroup; 
				this->numLocalThread = localThreadSize; 
			}

			void setUpdate(size_t _clUpdate) override
			{
				this->clUpdate = _clUpdate; 
			}

			void prepare(bool initDocs = true, size_t minWordCnt = 0, size_t removeTopN = 0) override
			{
				BaseClass::prepare(initDocs, minWordCnt, removeTopN);
				static_cast<DerivedClass*>(this)->prepareCL();
			}

			int train(size_t iteration, size_t numWorkers, ParallelScheme ps) override
			{
				ThreadPool pool(0);
				const size_t V = this->realV;
				auto& queue = this->clMgr.getQueue();
				size_t toIter = this->iterated + iteration;
				auto func = clMgr.getKernelFunc<uint32_t, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
					/*cl::LocalSpaceArg,*/ cl::Buffer, float, uint32_t, uint32_t>("trainLDA");
				while (this->iterated < toIter)
				{
					size_t oIter = toIter - this->iterated;
					if (this->optimInterval) oIter = std::min(oIter, this->optimInterval - (this->iterated % this->optimInterval));
					try
					{
						while (oIter > 0)
						{
							size_t iter = std::min(oIter, this->clUpdate);
							func(cl::EnqueueArgs(clMgr.getQueue(), cl::NDRange(numGroup * numLocalThread), cl::NDRange(numLocalThread)),
								this->docs.size(), clBufWsOffset, clBufWs, clBufVDOffset, clBufVPartition, clBufZs, clBufNumByTopicDoc, clBufNumByWordTopic, clBufNumByTopic,
								clBufAlphas, this->eta, iter, this->rg()).wait();
							this->iterated += iter;
							oIter -= iter;
						}
					}
					catch (const cl::Error& e)
					{
						std::cerr << OpenCLMgr::translateOpenCLError(e.err()) << " in " << e.what();
						throw;
					}
					//printf("Run In GPU: %g ms\n", timer.getAndReset());

					static_cast<DerivedClass*>(this)->receiveSamplingResult(queue);
					//printf("CGS on GPU: %g ms\n", timer.getAndReset());

					if (this->iterated >= this->burnIn && this->optimInterval && this->iterated % this->optimInterval == 0)
					{
						static_cast<DerivedClass*>(this)->optimizeParameters(pool, &this->globalState, nullptr);
					}
				}
				return 0;
			}

			template<bool _Together, ParallelScheme _ps, typename _Iter>
			std::vector<double> _infer(_Iter docFirst, _Iter docLast, size_t maxIter, FLOAT tolerance, size_t numWorkers) const
			{
				return {};
			}

		};
	}
}