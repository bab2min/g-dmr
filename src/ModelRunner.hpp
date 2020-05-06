#pragma once
#include <string>
#include <vector>
#include <tuple>
#include <chrono>
#include <iostream>
#include <fstream>

#include "Utils/Utils.hpp"
#include "TopicModel/TopicModel.hpp"
#include "TopicModel/LDAModel.hpp"
#include "TopicModel/DMRModel.hpp"
#include "TopicModel/GDMRModel.hpp"

#ifdef OPENCL
#include "OpenCL/TopicModel/LDAModel.hpp"
#include "OpenCL/TopicModel/DMRModel.hpp"
#include "OpenCL/TopicModel/GDMRModel.hpp"
#endif

using namespace std;

struct ModelRunnerBase
{
	struct Args
	{
		string modelType, load, save, bestSave, inference, stopword;
		vector<string> input;
		string saveTopicAssign, bestSaveTopicAssign, saveParameters, saveWordDist, saveTopicDistByDoc;
		tomoto::TermWeight weight = tomoto::TermWeight::one;
		size_t saveFullModel = 0;
		size_t worker = 0;
		size_t K = 1;
		double alpha = 0, eta = 0.01, gamma = 0.1, sigma = 1.0, lambda = 0.1;
		double alphaEps = 1e-10, sigma0 = 3.0;
		size_t seed = 0;
		size_t numMDFields = 0;
		size_t degree = 1;
		vector<size_t> degrees;
		int verbose = 2;
		size_t repeat = 1;
		size_t optimInterval = -1, bi = 0, optimRepeat = 5;
		size_t dirichletEstIteration = -1;
		size_t update = 10;
		size_t iteration = 1000;
		size_t numMaxLine = -1, numMaxLength = -1;
		size_t numTopWords = 25, numTopWordsSaving = -1;
		size_t saveUpdateInterval = 100;
		vector<float> mdMin, mdMax;
		int fa = 0, minCount = 0, fI = 100, rm = 0;
		size_t clDevice = 0;
		size_t clGroup = 32, clLocal = 128, clUpdate = 10;
	};

	virtual ~ModelRunnerBase() {}
	virtual int run(Args _args) = 0;
};


tuple<double, double, double, double> getDescStat(const vector<double>& d)
{
	double avg = 0, sqAvg = 0, vMin = FLT_MAX, vMax = FLT_MIN;
	for (auto&e : d)
	{
		avg += e;
		sqAvg += e * e;
		vMin = min(e, vMin);
		vMax = max(e, vMax);
	}
	avg /= d.size();
	sqAvg /= d.size();
	return make_tuple(avg, sqrt(max(sqAvg - avg * avg, 0.)), vMin, vMax);
}


class Timer
{
public:
	chrono::high_resolution_clock::time_point point;
	Timer()
	{
		reset();
	}
	void reset()
	{
		point = chrono::high_resolution_clock::now();
	}

	double getElapsed() const
	{
		return chrono::duration <double, milli>(chrono::high_resolution_clock::now() - point).count();
	}

	double getAndReset()
	{
		double ret = getElapsed();
		reset();
		return ret;
	}
};


template<class _Derived, class _Model>
struct ModelRunner : public ModelRunnerBase
{
	using Model = _Model;
	static ModelRunnerBase* factory() { return new _Derived; }
	Args args;
	shared_ptr<_Model> model;
	unordered_set<string> stopwordSet;
	vector<size_t> loadedDocId;
	bool filterSentDelim(const string& str)
	{
		return !stopwordSet.count(str);
	}

	void loadStopwordSet()
	{
		if (args.stopword.empty()) return;
		ifstream ifs{ args.stopword };
		string line;
		while (getline(ifs, line))
		{
			while (!line.empty() && isspace(line.back()))  line.pop_back();
			stopwordSet.emplace(line);
		}
	}

#ifdef OPENCL
	void initCL(tomoto::ocl::ICL* model)
	{
		auto selectedDevice = model->selectDevice(args.clDevice - 1);
		if (selectedDevice.empty())
		{
			printf("Device %d doesn't exist.\n", args.clDevice);
			throw runtime_error{ "" };
		}
		model->setWorkSize(args.clGroup, args.clLocal);
		model->setUpdate(args.clUpdate);
		printf("OpenCL Device '%s' is selected. (Running on %zd Groups x %zd Local threads) %zd updates per work\n", selectedDevice.c_str(), args.clGroup, args.clLocal, args.clUpdate);
	}
#endif

	void trainModel(string path_suffix, double& perp, double& elapsed)
	{
		auto saveResult = [&]()
		{
			if (!args.saveTopicAssign.empty())
			{
				ofstream ofs{ args.saveTopicAssign + path_suffix };
				for (size_t d = 0; d < model->getNumDocs(); ++d)
				{
					static_cast<_Derived*>(this)->writeTopicAssign(ofs, *static_cast<const typename Model::DefaultDocType*>(model->getDoc(d)));
					ofs << endl;
				}
			}
			if (!args.saveTopicDistByDoc.empty())
			{
				ofstream ofs{ args.saveTopicDistByDoc + path_suffix };
				for (size_t d : loadedDocId)
				{
					if (d != (size_t)-1)
					{
						static_cast<_Derived*>(this)->writeTopicDist(ofs, *static_cast<const typename Model::DefaultDocType*>(model->getDoc(d)));
					}
					else
					{
						ofs << "-";
					}
					ofs << endl;
				}
			}
			if (!args.saveWordDist.empty())
			{
				ofstream ofs{ args.saveWordDist + path_suffix };
				static_cast<_Derived*>(this)->writeWordDist(ofs);
			}
		};

		Timer timer;
		if (args.verbose) printf("Training %zd docs (%zd vocabs, %zd words)...\n", model->getNumDocs(), model->getV(), model->getN());
		if (model->getNumDocs() <= 0) return;
		if (args.verbose > 3)
		{
			static_cast<_Derived*>(this)->printTopics();
		}
		double bestPerp = INFINITY;
		for (size_t n = 0; n < args.iteration; n += args.update)
		{
			if (args.saveUpdateInterval && n % args.saveUpdateInterval == 0)
			{
				saveResult();
			}

			model->train(args.update, args.worker, tomoto::ParallelScheme::partition);
			double llpw = model->getLLPerWord();
			double perp = exp(-llpw);
			if (args.verbose != 3)
			{
				if (args.verbose) printf("(%03zd) Perp : %e (%g)\n", n + args.update, perp, llpw);
				if (args.verbose > 1) static_cast<_Derived*>(this)->printParams();
				if (args.verbose > 3)
				{
					static_cast<_Derived*>(this)->printTopics();
				}
			}

			if (perp < bestPerp)
			{
				if (args.verbose == 3)
				{
					printf("(%03zd) Perp : %e (%g)\n", n + args.update, perp, llpw);
					static_cast<_Derived*>(this)->printParams();
				}
				bestPerp = perp;
				if (!args.bestSave.empty())
				{
					ofstream ofs{ args.bestSave + path_suffix };
					model->saveModel(ofs, !!args.saveFullModel);
				}
				if (!args.bestSaveTopicAssign.empty())
				{
					ofstream ofs{ args.bestSaveTopicAssign + path_suffix };
					for (size_t d = 0; d < model->getNumDocs(); ++d)
					{
						static_cast<_Derived*>(this)->writeTopicAssign(ofs, *static_cast<const typename Model::DefaultDocType*>(model->getDoc(d)));
						ofs << endl;
					}
				}
			}
			fflush(stdout);
		}

		printf("\n=== Result ===\n");
		perp = model->getPerplexity();
		elapsed = timer.getElapsed();
		printf("Elapsed: %g ms, Perp: %e\n", elapsed, perp);
		fflush(stdout);
		if (!args.save.empty())
		{
			ofstream ofs{ args.save + path_suffix, ios_base::binary };
			model->saveModel(ofs, !!args.saveFullModel);
		}
		saveResult();
	}

	int run(Args _args) override
	{
		args = _args;
		if (args.load.empty())
		{
			printf("Input file = %s, ", tomoto::text::join(args.input.begin(), args.input.end()).c_str());
		}
		else
		{
			printf("Load model file = %s, ", args.load.c_str());
		}
		printf("Repeat %zd times, Print %zd top words, Using %zd workers", args.repeat, args.numTopWords, args.worker ? args.worker : thread::hardware_concurrency());
		if (args.seed) printf(",  Fixed random seed %zd", args.seed);
		printf("\n%zd iterations\n", args.iteration);
		if (!args.save.empty()) printf("Save model to file '%s'...\n", args.save.c_str());
		fflush(stdout);
		static const char* twMsg[] = { "one", "idf", "pmi", "idf_one" };
		vector<double> elapsed(args.repeat), perp(args.repeat);
		for (size_t i = 0; i < args.repeat; ++i)
		{
			string path_suffix;
			if (args.repeat > 1) path_suffix = tomoto::text::format(".%02d", i);
			if (!args.load.empty())
			{
				args.weight = (tomoto::TermWeight)0;
				if (args.verbose) printf("Loading binary model '%s'...\n", args.load.c_str());
			}

			do
			{
				model = shared_ptr<_Model>(static_cast<_Derived*>(this)->init(args.seed ? args.seed : random_device{}()));
				static_cast<_Derived*>(this)->setParameters();
				if (args.load.empty())
				{
					if (i == 0)
					{
						puts(static_cast<_Derived*>(this)->getParameterDesc().c_str());
						printf("Term Weighting: %s\n", twMsg[(int)args.weight]);
					}
					if (args.verbose) printf("Loading '%s'...\n", tomoto::text::join(args.input.begin(), args.input.end()).c_str());
					int numLine = load();
					if (!numLine)
					{
						printf("Wrong Input '%s'\n", tomoto::text::join(args.input.begin(), args.input.end()).c_str());
						return -1;
					}
					model->prepare(true, args.minCount, args.rm);
					if (args.verbose && args.minCount) printf("Min Count of Words: %d\n", args.minCount);
					if (args.verbose && args.numMaxLength != (size_t)-1) printf("Max Length of Document: %zd\n", args.numMaxLength);
					if (args.verbose && args.rm)
					{
						printf("Removed Top N words: ");
						size_t last = model->getVocabDict().size();
						for (size_t rmV = last - args.rm; rmV < last; ++rmV)
						{
							printf("%s, ", model->getVocabDict().toWord(rmV).c_str());
						}
						printf("\n");
					}
					trainModel(path_suffix, perp[i], elapsed[i]);
				}
				else
				{
					try
					{
						ifstream ifs{ args.load, ios_base::binary };
						model->loadModel(ifs);
						static_cast<_Derived*>(this)->onModelLoaded();
					}
					catch (const tomoto::serializer::UnfitException& e)
					{
						args.weight = (tomoto::TermWeight)((size_t)args.weight + 1);
						model.reset();
						continue;
					}
					loadedDocId.resize(model->getNumDocs());
					iota(loadedDocId.begin(), loadedDocId.end(), 0);
					puts(static_cast<_Derived*>(this)->getParameterDesc().c_str());
					printf("Term Weighting: %s\n", twMsg[(int)model->getTermWeight()]);
					trainModel(path_suffix, perp[i], elapsed[i]);
				}
				break;
			} while (args.weight < tomoto::TermWeight::size);

			if (!model)
			{
				printf("Failed to load binary model '%s'", args.load.c_str());
				return -1;
			}

			static_cast<_Derived*>(this)->printParams();
			fflush(stdout);
			if (!args.saveParameters.empty())
			{
				ofstream ofs{ args.saveParameters + path_suffix };
				static_cast<_Derived*>(this)->writeParams(ofs);
			}
			if(args.numTopWords) static_cast<_Derived*>(this)->printTopics();
			fflush(stdout);
		}

		printf("\n============\n");
		fflush(stdout);
		auto stat = getDescStat(elapsed);
		if (args.load.empty()) printf("Elapsed Stat\nAvg:%g ms, Std:%g ms, Min:%g ms, Max:%g ms\n", get<0>(stat), get<1>(stat), get<2>(stat), get<3>(stat));
		stat = getDescStat(perp);
		printf("Perplexity Stat\nAvg:%g, Std:%g, Min:%g, Max:%g\n", get<0>(stat), get<1>(stat), get<2>(stat), get<3>(stat));

		if (args.inference.empty()) return 0;
		printf("Inference unseen document file '%s'...\n", args.inference.c_str());
		ifstream inf{ args.inference };
		string line;
		size_t numLine = 0;

		vector<unique_ptr<tomoto::DocumentBase>> docs;
		vector<tomoto::DocumentBase*> pdocs;
		while (getline(inf, line))
		{
			unique_ptr<tomoto::DocumentBase> doc;
			if (static_cast<_Derived*>(this)->loadLine(line, &doc))
			{
				numLine++;
				docs.emplace_back(move(doc));
				pdocs.emplace_back(docs.back().get());
			}
		}
		static_cast<_Derived*>(this)->infer(pdocs, args.fa);
		return 0;
	}

	int load()
	{
		loadStopwordSet();
		string line;
		size_t numLine = 0;
		for (auto& p : args.input)
		{
			ifstream f{ p };
			while (getline(f, line) && numLine < args.numMaxLine)
			{
				if (static_cast<_Derived*>(this)->loadLine(line))
				{
					loadedDocId.emplace_back(numLine++);
				}
				else
				{
					loadedDocId.emplace_back(-1);
				}
			}
		}
		return numLine;
	}

	void onModelLoaded()
	{

	}

	void writeTopicDist(ostream& out, const typename Model::DefaultDocType& doc)
	{
		for (auto t : model->getTopicsByDoc(&doc))
		{
			out << t << '\t';
		}
	}
};

template<typename _Derived = void,
	typename _Model = tomoto::ILDAModel>
struct LDARunner : public ModelRunner<
	typename conditional<is_same<_Derived, void>::value, LDARunner<>, _Derived>::type,
	_Model>
{
	using Model = _Model;
	Model* init(size_t seed)
	{
#ifdef OPENCL
		if (this->args.clDevice)
		{
			auto ret = new tomoto::ocl::CL_LDAModel<tomoto::TermWeight::one>(
				this->args.K, (tomoto::FLOAT)this->args.alpha, (tomoto::FLOAT)this->args.eta, tomoto::RandGen{ seed });
			this->initCL(ret);
			return ret;
		}
#endif
		return Model::create(this->args.weight, this->args.K, (tomoto::FLOAT)this->args.alpha, (tomoto::FLOAT)this->args.eta, tomoto::RandGen{seed});

	}

	void setParameters()
	{
		if (this->args.optimInterval != (size_t)-1) this->model->setOptimInterval(this->args.optimInterval);
		this->model->setBurnInIteration(this->args.bi);
	}

	int loadLine(const string& line, unique_ptr<tomoto::DocumentBase>* doc = nullptr)
	{
		istringstream iss{ line };
		auto begin = istream_iterator<string>{ iss }, end = istream_iterator<string>{};
		for (size_t i = 0; i < this->args.numMDFields; ++i)
		{
			if (begin == end) break;
			++begin;
		}
		vector<string> words;
		for ( ; begin != end; ++begin)
		{
			if (this->filterSentDelim(*begin)) words.emplace_back(*begin);
			if (words.size() >= this->args.numMaxLength) break;
		}
		if (!doc)
		{
			this->model->addDoc(words);
			return words.size();
		}
		else
		{
			*doc = this->model->makeDoc(words);
			return (*doc)->words.size();
		}
	}

	string getParameterDesc()
	{
		return tomoto::text::format("LDA model\nK = %zd, alpha = %g, eta = %g", this->model->getK(), this->model->getAlpha(), this->model->getEta());
	}

	void printParams()
	{

	}

	void printTopics()
	{
		auto counts = this->model->getCountByTopic();
		for (size_t k = 0; k < this->model->getK(); ++k)
		{
			printf("== Topic %zd == (%zd) \n", k, counts[k]);
			for (auto& t : this->model->getWordsByTopicSorted(k, this->args.numTopWords))
			{
				printf("\t%s\t%.3g\t%zd\n", t.first.c_str(), t.second, (size_t)(counts[k] * t.second + 0.5));
			}
		}
	}

	void infer(const vector<tomoto::DocumentBase*>& docs, bool together)
	{
		vector<double> lls = this->model->infer(docs, this->args.fI, -1, this->args.worker, 
			tomoto::ParallelScheme::partition, together);
		size_t totSize = 0;
		double ll = together ? lls[0] : 0;
		size_t i = 0;
		for (auto& doc : docs)
		{
			totSize += doc->words.size();
			if (!together)
			{
				printf("LL: %g (%g) %g\t", lls[i], lls[i] / doc->words.size(), exp(-lls[i] / doc->words.size()));
				ll += lls[i++];
			}
			printf("[");
			for (auto& t : this->model->getTopicsByDoc(doc))
			{
				printf("%.3f, ", t);
			}
			printf("]\n");
		}
		printf("LL: %g (%g) %g\n", ll, ll / totSize, exp(-ll / totSize));
	}

	void writeTopicAssign(ostream& out, const typename Model::DefaultDocType& doc)
	{
		for (size_t w = 0; w < doc.words.size(); ++w)
		{
			size_t t = doc.wOrder.empty() ? w : doc.wOrder[w];
			out << this->model->getVocabDict().toWord(doc.words[t]) << '/' << (int)doc.Zs[t] << ' ';
		}
	}

	void writeWordDist(ostream& ostr)
	{
		for (size_t k = 0; k < this->model->getK(); ++k)
		{
			ostr << k << endl;
			for (auto& t : this->model->getWordsByTopicSorted(k, this->args.numTopWordsSaving))
			{
				ostr << t.first << '\t' << t.second << endl;
			}
			ostr << endl;
		}
	}

	void writeParams(ostream& ostr)
	{
		ostr << "alpha\t" << this->model->getAlpha() << endl;
		ostr << "eta\t" << this->model->getEta() << endl;
	}
};

template<typename _Derived = void,
	typename _Model = tomoto::IDMRModel>
struct DMRRunner : public LDARunner<
	typename conditional<is_same<_Derived, void>::value, DMRRunner<>, _Derived>::type,
	_Model>
{
	using Model = _Model;
	Model* init(size_t seed)
	{
#ifdef OPENCL
		if (this->args.clDevice)
		{
			auto ret = new tomoto::ocl::CL_DMRModel<tomoto::TermWeight::one>(
				this->args.K, (tomoto::FLOAT)this->args.alpha, (tomoto::FLOAT)this->args.sigma,(tomoto::FLOAT)this->args.eta, 
				(tomoto::FLOAT)this->args.alphaEps, tomoto::RandGen{ seed });
			this->initCL(ret);
			ret->setOptimRepeat(this->args.optimRepeat);
			return ret;
		}
#endif
		auto ret = Model::create( this->args.weight, this->args.K, 
			(tomoto::FLOAT)this->args.alpha, (tomoto::FLOAT)this->args.sigma, (tomoto::FLOAT)this->args.eta, 
			this->args.alphaEps, tomoto::RandGen{ seed } );
		ret->setOptimRepeat(this->args.optimRepeat);
		return ret;
	}

	int loadLine(const string& line, unique_ptr<tomoto::DocumentBase>* doc = nullptr)
	{
		istringstream iss{ line };
		auto begin = istream_iterator<string>{ iss }, end = istream_iterator<string>{};
		vector<string> features;
		for (size_t i = 0; i < this->args.numMDFields; ++i)
		{
			if (begin == end) break;
			features.emplace_back(*begin++);
		}
		vector<string> words;
		for (; begin != end; ++begin)
		{
			if (this->filterSentDelim(*begin)) words.emplace_back(*begin);
			if (words.size() >= this->args.numMaxLength) break;
		}
		if (!doc)
		{
			return this->model->addDoc(words, features);
		}
		else
		{
			*doc = this->model->makeDoc(words, features);
			return (*doc)->words.size();
		}
	}

	string getParameterDesc()
	{
		return tomoto::text::format("DMR model\nK = %zd, default alpha = %g, sigma = %g, eta = %g", 
			this->args.K, this->args.alpha, this->args.sigma, this->args.eta);
	}

	void printParams()
	{
		printf("== Parameters ==\n");
		for (size_t f = 0; f < this->model->getF(); ++f)
		{
			printf("%s: ", this->model->getMetadataDict().toWord(f).c_str());
			for (auto& p : this->model->getLambdaByMetadata(f))
			{
				printf("%g, ", p);
			}
			printf("\n");
		}
	}

	void writeParams(ostream& ostr)
	{
		ostr << "alpha\t" << this->model->getAlpha() << endl;
		ostr << "sigma\t" << this->model->getSigma() << endl;
		ostr << "eta\t" << this->model->getEta() << endl;
		for (size_t f = 0; f < this->model->getF(); ++f)
		{
			ostr << "alpha\t" << this->model->getMetadataDict().toWord(f);
			for (auto& p : this->model->getLambdaByMetadata(f))
			{
				ostr << '\t' << p;
			}
			ostr << endl;
		}
	}
};

template<typename _Derived = void,
	typename _Model = tomoto::IGDMRModel>
	struct GDMRRunner : public DMRRunner<
	typename conditional<is_same<_Derived, void>::value, GDMRRunner<>, _Derived>::type,
	_Model>
{
	using BaseClass = LDARunner<
		typename conditional<is_same<_Derived, void>::value, GDMRRunner<>, _Derived>::type,
		_Model>;
	using Model = _Model;
	Model* init(size_t seed)
	{
		if (this->args.degrees.size() != this->args.numMDFields)
		{
			throw invalid_argument("length of -D is different from -F");
		}

#ifdef OPENCL
		if (this->args.clDevice)
		{
			auto ret = new tomoto::ocl::CL_GDMRModel<tomoto::TermWeight::one>(
				this->args.K, this->args.degrees, 
				(tomoto::FLOAT)this->args.alpha, (tomoto::FLOAT)this->args.sigma, (tomoto::FLOAT)this->args.eta,
				(tomoto::FLOAT)this->args.alphaEps, tomoto::RandGen{ seed });
			this->initCL(ret);
			ret->setMdRange(this->args.mdMin, this->args.mdMax);
			ret->setOptimRepeat(this->args.optimRepeat);
			ret->setSigma0(this->args.sigma0);
			return ret;
		}
#endif
		auto ret = Model::create(this->args.weight, this->args.K, this->args.degrees,
			(tomoto::FLOAT)this->args.alpha, (tomoto::FLOAT)this->args.sigma, (tomoto::FLOAT)this->args.eta,
			this->args.alphaEps, tomoto::RandGen{ seed });
		ret->setMdRange(this->args.mdMin, this->args.mdMax);
		ret->setOptimRepeat(this->args.optimRepeat);
		ret->setSigma0(this->args.sigma0);
		return ret;
	}

	string getParameterDesc()
	{
		return tomoto::text::format("g-DMR Legendre model\nK = %zd, metadata fields = %zd, degrees = (%s), default alpha = %g, alphaEps = %g, sigma = %g, eta = %g",
			this->model->getK(), this->model->getFs().size(), tomoto::text::join(this->model->getFs().begin(), this->model->getFs().end()).c_str(),
			this->model->getAlpha(), this->model->getAlphaEps(), this->model->getSigma(), this->model->getEta());
	}

	void printParams()
	{
		printf("== Parameters ==\n");
		for (size_t k = 0; k < this->model->getK(); ++k)
		{
			for (auto& p : this->model->getLambdaByTopic(k))
			{
				printf("%g, ", p);
			}
			printf("\n");
		}
	}

	void writeParams(ostream& ostr)
	{
		ostr << "alpha\t" << this->model->getAlpha() << endl;
		ostr << "sigma\t" << this->model->getSigma() << endl;
		ostr << "eta\t" << this->model->getEta() << endl;
		for (size_t k = 0; k < this->model->getK(); ++k)
		{
			ostr << "lambda";
			for (auto& p : this->model->getLambdaByTopic(k))
			{
				ostr << '\t' << p;
			}
			ostr << endl;
		}
	}
};
