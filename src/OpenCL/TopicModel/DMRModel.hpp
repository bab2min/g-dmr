#pragma once
#include "LDAModel.hpp"
#include "../../TopicModel/DMRModel.hpp"

/*
Implementation of LDA using Parallel Collapsed Gibbs Sampling on GPU by bab2min

* Yan, F., Xu, N., & Qi, Y. (2009). Parallel inference for latent dirichlet allocation on graphics processing units. In Advances in Neural Information Processing Systems (pp. 2134-2142).
*/

namespace tomoto
{
	namespace ocl
	{
		template<TermWeight _TW,
			typename _Interface = IDMRModel,
			typename _Derived = void,
			typename _DocType = DocumentDMR<_TW, flags::continuous_doc_data>,
			typename _ModelState = ModelStateDMR<_TW>,
			template<TermWeight, size_t,
			typename ...> class _BaseModel = DMRModel
		>
		class CL_DMRModel : public CL_LDAModel<_TW, _Interface,
			typename std::conditional<std::is_same<_Derived, void>::value, CL_DMRModel<_TW>, _Derived>::type,
			_DocType, _ModelState, _BaseModel>
		{
		protected:
			using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, CL_DMRModel<_TW>, _Derived>::type;
			using BaseClass = CL_LDAModel<_TW, _Interface, DerivedClass, _DocType, _ModelState, _BaseModel>;
			friend BaseClass;
			friend typename BaseClass::BaseClass;
			friend typename BaseClass::BaseClass::BaseClass;

			static constexpr const char* TRAIN_FUNC = "trainDMR";
			static constexpr const char* OBJ_FUNC = "calcObjDMR";

			cl::Buffer clBufFByDoc, clBufLambdas, clBufOutput;

			size_t getNumFeatures() const
			{
				return this->F;
			}

			std::vector<uint32_t> getMetadataVector() const
			{
				std::vector<uint32_t> Fs;
				for (auto& doc : this->docs)
				{
					Fs.emplace_back(doc.metadata);
				}
				return Fs;
			}

			void initializeBuffer()
			{
				BaseClass::initializeBuffer();

				auto Fs = static_cast<DerivedClass*>(this)->getMetadataVector();
				const auto F = static_cast<DerivedClass*>(this)->getNumFeatures();

				clBufFByDoc = cl::Buffer(this->clMgr.getContext(), CL_MEM_READ_ONLY, sizeof(typename decltype(Fs)::reference) * Fs.size());
				clBufLambdas = cl::Buffer(this->clMgr.getContext(), CL_MEM_READ_ONLY, sizeof(float) * F * this->K);
				clBufOutput = cl::Buffer(this->clMgr.getContext(), CL_MEM_WRITE_ONLY, sizeof(float) * (F * this->K + 1) * this->numGroup);
				auto& queue = this->clMgr.getQueue();
				queue.enqueueWriteBuffer(clBufFByDoc, true, 0, sizeof(typename decltype(Fs)::reference) * Fs.size(), &Fs[0]);
				queue.enqueueWriteBuffer(clBufLambdas, true, 0, sizeof(float) * this->K * F, this->lambda.data());
			}

			void prepareCL()
			{
				const size_t V = this->realV;
				std::string err;
				if (this->clMgr.buildProgramFromFile({ "clCode/Common.cl", "clCode/mwc64x.cl", "clCode/DMR.cl" }, "",
					text::format("-D TMT_K=%d -D TMT_V=%d -D F=%d -D TMT_ALPHA=%e -D TMT_ALPHA_EPS=%e ",
						this->K, V, this->F, this->alpha, this->alphaEps), &err))
				{
					std::cerr << err;
					throw cl::Error(-1);
				}

				initializeBuffer();
			}

			void optimizeParameters(ThreadPool& pool, _ModelState* localData, RandGen* rgs)
			{
				BaseClass::optimizeParameters(pool, localData, rgs);
				auto& queue = this->clMgr.getQueue();
				static_cast<DerivedClass*>(this)->sendLambdas(queue, this->lambda.data());
			}

			FLOAT evaluateLambdaObj(Eigen::Ref<Eigen::Matrix<FLOAT, -1, 1>> x, Eigen::Matrix<FLOAT, -1, 1>& g, ThreadPool& pool, _ModelState* localData) const
			{
				if ((x.array() > this->maxLambda).any()) return INFINITY;

				FLOAT fx = static_cast<const DerivedClass*>(this)->getNegativeLambdaLL(x, g);
				const auto F = static_cast<const DerivedClass*>(this)->getNumFeatures();
				auto& queue = this->clMgr.getQueue();
				try
				{
					queue.enqueueFillBuffer(clBufOutput, 0.f, 0, sizeof(float) * (F * this->K + 1) * this->numGroup);
				}
				catch (const cl::Error& e)
				{
					std::cerr << OpenCLMgr::translateOpenCLError(e.err()) << " in " << e.what();
					throw;
				}
				static_cast<const DerivedClass*>(this)->sendLambdas(queue, x.data());

				try
				{
					auto func = this->clMgr.template getKernelFunc<uint32_t, cl::Buffer, cl::Buffer,
						float, cl::Buffer, cl::Buffer, cl::Buffer/*, cl::Buffer*/>(static_cast<const DerivedClass*>(this)->OBJ_FUNC);
					func(cl::EnqueueArgs(this->clMgr.getQueue(), 
						cl::NDRange(this->numGroup * this->numLocalThread), 
						cl::NDRange(this->numLocalThread)),
						this->docs.size(), this->clBufWsOffset, this->clBufNumByTopicDoc, 
						this->sigma, clBufFByDoc, clBufLambdas, clBufOutput/*, clBufDebug*/).wait();
				}
				catch (const cl::Error& e)
				{
					std::cerr << OpenCLMgr::translateOpenCLError(e.err()) << " in " << e.what();
					throw;
				}

				Eigen::MatrixXf result(this->K * F + 1, this->numGroup);
				try
				{
					queue.enqueueReadBuffer(clBufOutput, true, 0, sizeof(float) * result.size(), result.data());
				}
				catch (const cl::Error& e)
				{
					std::cerr << OpenCLMgr::translateOpenCLError(e.err()) << " in " << e.what();
					throw;
				}

				fx += result.row(0).sum();
				if (fx < 0 || !std::isfinite(fx)) return INFINITY;
				g += result.block(1, 0, this->K * F, this->numGroup).rowwise().sum();
				return fx;
			}

			void sendLambdas(cl::CommandQueue& queue, const float* data) const
			{
				try
				{
					queue.enqueueWriteBuffer(clBufLambdas, true, 0, sizeof(float) * this->K * static_cast<const DerivedClass*>(this)->getNumFeatures(), data);
				}
				catch (const cl::Error& e)
				{
					std::cerr << OpenCLMgr::translateOpenCLError(e.err()) << " in " << e.what();
					throw;
				}
			}

		public:
			using BaseClass::BaseClass;

			int train(size_t iteration, size_t numWorkers, ParallelScheme ps) override
			{
				ThreadPool pool(1);
				size_t toIter = this->iterated + iteration;
				const size_t V = this->realV;
				auto& queue = this->clMgr.getQueue();
				auto func = this->clMgr.template getKernelFunc<uint32_t, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
					cl::Buffer, float, cl::Buffer, uint32_t, uint32_t>(static_cast<DerivedClass*>(this)->TRAIN_FUNC);
				while (this->iterated < toIter)
				{
					try
					{
						size_t oIter = toIter - this->iterated;
						if (this->optimInterval) oIter = std::min(oIter, this->optimInterval - (this->iterated % this->optimInterval));
						try
						{
							while (oIter > 0)
							{
								size_t iter = std::min(oIter, this->clUpdate);
								func(cl::EnqueueArgs(this->clMgr.getQueue(),
									cl::NDRange(this->numGroup * this->numLocalThread), cl::NDRange(this->numLocalThread)),
									this->docs.size(), this->clBufWsOffset, this->clBufWs, this->clBufVDOffset, this->clBufVPartition,
									this->clBufZs, this->clBufNumByTopicDoc, this->clBufNumByWordTopic, this->clBufNumByTopic,
									clBufLambdas, this->eta, clBufFByDoc, iter, this->rg()).wait();
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
					catch (const exception::TrainingError& e)
					{
						std::cerr << e.what() << std::endl;
						int ret = static_cast<DerivedClass*>(this)->restoreFromTrainingError(e, pool,
							&this->globalState, &this->rg);
						if (ret < 0) return ret;
					}
				}
				return 0;
			}
		};
	}
}