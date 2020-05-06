#pragma once
#include "DMRModel.hpp"
#include "../../TopicModel/GDMRModel.hpp"

/*
Implementation of LDA using Parallel Collapsed Gibbs Sampling on GPU by bab2min

* Yan, F., Xu, N., & Qi, Y. (2009). Parallel inference for latent dirichlet allocation on graphics processing units. In Advances in Neural Information Processing Systems (pp. 2134-2142).
*/
namespace tomoto
{
	namespace ocl
	{
		template<TermWeight _TW,
			typename _Interface = IGDMRModel,
			typename _Derived = void,
			typename _DocType = DocumentGDMR<_TW, flags::continuous_doc_data>,
			typename _ModelState = ModelStateGDMR<_TW>,
			template<TermWeight, size_t,
			typename ...> class _BaseModel = GDMRModel
		>
			class CL_GDMRModel : public CL_DMRModel<_TW, _Interface,
			typename std::conditional<std::is_same<_Derived, void>::value, CL_GDMRModel<_TW>, _Derived>::type,
			_DocType, _ModelState, _BaseModel>
		{
		protected:
			using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, CL_GDMRModel<_TW>, _Derived>::type;
			using BaseClass = CL_DMRModel<_TW, _Interface, DerivedClass, _DocType, _ModelState, _BaseModel>;
			friend BaseClass;
			friend typename BaseClass::BaseClass;
			friend typename BaseClass::BaseClass::BaseClass;
			friend typename BaseClass::BaseClass::BaseClass::BaseClass;
			friend typename BaseClass::BaseClass::BaseClass::BaseClass::BaseClass;

			static constexpr const char* TRAIN_FUNC = "trainGDMR";
			static constexpr const char* OBJ_FUNC = "calcObjGDMR";

			std::vector<float> getMetadataVector() const
			{
				std::vector<float> Fs;
				for (auto& doc : this->docs)
				{
					Fs.insert(Fs.end(), doc.metadataC.begin(), doc.metadataC.end());
				}
				return Fs;
			}

			void prepareCL()
			{
				const size_t V = this->realV;
				std::string err;
				std::string opt = text::format("-D TMT_K=%d -D TMT_V=%d -D F=%d -D TMT_F_STRIDE=%d -D TMT_ALPHA=%e -D TMT_ALPHA_EPS=%e ", 
					this->K, V, this->getNumFeatures(), this->degreeByF.size(), this->alpha, this->alphaEps);

				std::vector<size_t> digit(this->degreeByF.size());
				std::string source;
				size_t maxOrder = *std::max_element(this->degreeByF.begin(), this->degreeByF.end());
				for (size_t i = 0; i <= maxOrder; ++i)
				{
					std::string s = std::to_string(slp::slpGetCoef(i, i));
					for (size_t j = 1; j <= i; ++j)
					{
						s = "(" + s + ") * x + " + std::to_string(slp::slpGetCoef(i, i - j));
					}
					source += text::format("float slp_%d(float x)\n{\n\treturn %s;\n}\n\n", i, s.c_str());
				}

				source += "void getTerms(constant float* Fs, float* ret)\n{\n";
				for (size_t i = 0; i < this->F; ++i)
				{
					source += "*ret++ = ";
					for (size_t j = 0; j < this->degreeByF.size(); ++j)
					{
						if (j) source += " * ";
						source += text::format("slp_%d(Fs[%d])", digit[j], j);
					}
					source += ";\n";

					for (size_t u = 0; u < digit.size() && ++digit[u] > this->degreeByF[u]; ++u)
					{
						digit[u] = 0;
					}
				}
				source += "}\n\n";

				if (this->clMgr.buildProgramFromFile({ "clCode/Common.cl", "clCode/mwc64x.cl", "clCode/GDMR.cl" }, source, opt, &err))
				{
					std::cerr << err;
					throw cl::Error(-1);
				
				}
				this->initializeBuffer();
			}

		public:
			using BaseClass::BaseClass;

		};
	}
}