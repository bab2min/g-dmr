#include <locale> 
#include <codecvt>
#include "cxxopts.hpp"

#ifdef _WIN32
#define OPENCL
#define NOMINMAX
#include <Windows.h>
#endif

#include "ModelRunner.hpp"


using namespace std;

template<typename _Ty>
vector<_Ty> stringToVector(const string& str)
{
	vector<_Ty> ret;
	try
	{
		stringstream ss{ str };
		string num;
		while (getline(ss, num, ','))
		{
			_Ty ty;
			stringstream t{ num };
			if(t >> ty) ret.emplace_back(ty);
		}
	}
	catch (const exception&) {}
	return ret;
}

using model_pair_t = std::pair<const char*, ModelRunnerBase*(*)()>;

initializer_list<model_pair_t> modelTypes =
{
	{ "lda", LDARunner<>::factory },
	{ "dmr", DMRRunner<>::factory },
	{ "gdmr", GDMRRunner<>::factory },
};

unique_ptr<ModelRunnerBase> newInstance(const string & modelType)
{
	auto it = find_if(modelTypes.begin(), modelTypes.end(), 
		[&modelType](const model_pair_t& it)
		{ 
		return it.first == modelType; 
		}
	);
	if(it == modelTypes.end() || !it->second) return nullptr;
	return unique_ptr<ModelRunnerBase>(it->second());
}

int main(int argc, char* argv[])
{
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
	setvbuf(stdout, nullptr, _IONBF, 0);
#endif
	ModelRunnerBase::Args args;
	bool clList = false;
	try
	{
		cxxopts::Options options("tomoto", "Topic Model Toolkit for C++");
		options
			.positional_help("[model input]")
			.show_positional_help();

		options.add_options()
			("model", "Topic Model", cxxopts::value<std::string>(), accumulate(modelTypes.begin(), modelTypes.end(), string{}, 
				[](const string& a, const model_pair_t& p) -> string
				{
				return a + (a.empty() ? "" : ", ") + p.first;
				}
			))
			("tw", "Term Weighting", cxxopts::value<std::string>()->default_value("one"), "one, idf, pmi, idf_one; (default = one)")
			("i,input", "Input File", cxxopts::value<std::vector<std::string>>(), "Input file pathes that contains documents per line")
			("maxline", "Number of Lines to be read ", cxxopts::value<int>())
			("maxlen", "Max number of words for each document", cxxopts::value<int>())
			("l,load", "Load Model File", cxxopts::value<std::string>(), "Model file path to be loaded")
			("v,save", "Save Model File", cxxopts::value<std::string>(), "Model file path to be saved")
			("tsave", "Save Topic Assignment into file", cxxopts::value<std::string>())
			("btsave", "Save Best Topic Assignment into file", cxxopts::value<std::string>())
			("fullmodel", "Full Model Save", cxxopts::value<int>()->implicit_value("1"))
			("bsave", "Save Best-Fitted Model File", cxxopts::value<std::string>(), "If this parameter is declared, the best fitted model would be save in this path.")
			("wsave", "Save Word Distribution into file", cxxopts::value<std::string>())
			("psave", "Save Paramters into file", cxxopts::value<std::string>())
			("dsave", "Save Topic Distribution by docs into file", cxxopts::value<std::string>())
			("V,verbose", "Verbose", cxxopts::value<int>(), "0 = print nothing, 1 = print only perp., 2 = print perp. & params")
			("h,help", "Help")
			("version", "Version")
			("mc", "Minimum Count of Words", cxxopts::value<int>())
			("rm", "Remove Top N Words", cxxopts::value<int>())
			("stopword", "Stopword File", cxxopts::value<std::string>())
			
			("w,worker", "Number of Workes", cxxopts::value<int>(), "The number of workers(std::thread) for inferencing model, default value is 0 which means the number of cores in system")
			("S,seed", "Seed of Random", cxxopts::value<int>(), "The seed value of random generator. Default value is 0 which means device_random, and other values generate fixed random numbers. Although the seed value is identical, the result could be different due to multithreading race if the number of workers is greater than 1.")
			("K,topic", "Number of Topics", cxxopts::value<int>())
			("F,mdfield", "Number of Metadata Fields", cxxopts::value<int>())
			("D,degree", "Number of Degree (g-DMR)", cxxopts::value<std::string>())
			("words", "Number of Top Words", cxxopts::value<int>())
			("ws", "Number of Top Words to be saved (with --wsave option)", cxxopts::value<int>())
			("bi", "Burn-in Iteration", cxxopts::value<int>())
			("oi", "Optimizing Interval", cxxopts::value<int>())
			("dei", "Dirichlet Estimation Iteration", cxxopts::value<int>())
			("r,repeat", "Number of Repeats", cxxopts::value<int>())
			("I,iteration", "Iterations", cxxopts::value<int>())
			("update", "Interval of update", cxxopts::value<int>())
			("mdm", "Metadata Minimum (g-DMR)", cxxopts::value<string>())
			("mdM", "Metadata Maximum (g-DMR)", cxxopts::value<string>())
			
			("a,alpha", "Alpha", cxxopts::value<double>())
			("e,eta", "Eta", cxxopts::value<double>())
			("s,sigma", "Sigma", cxxopts::value<double>())
			("s0", "Sigma0", cxxopts::value<double>())
			("alphaEps", "Alpha Epsilon", cxxopts::value<double>())
			
			("f,inference", "Inference File", cxxopts::value<std::string>(), "File path that would be inferenced by fitted model")
			("fa", "Inference all together", cxxopts::value<int>()->implicit_value("1"), "")
			("fI", "Iteration for Inference", cxxopts::value<int>())

#ifdef OPENCL
			("cl", "OpenCL device name for GPU accelerating", cxxopts::value<int>())
			("clGroup", "Number of OpenCL work groups", cxxopts::value<int>())
			("clLocal", "Number of OpenCL local threads", cxxopts::value<int>())
			("clUpdate", "Max Number of iteration for an OpenCL work", cxxopts::value<int>())
			("clList", "")
#endif
			;

		options.parse_positional({ "model", "input"});

		try 
		{

			auto result = options.parse(argc, argv);

			if (result.count("version"))
			{
				cout << "v0.1" << endl;
				return 0;
			}
			if (result.count("help"))
			{
				cout << options.help({ "" }) << endl;
				return 0;
			}
#ifdef OPENCL
			if ((clList = result.count("clList")))
			{
				goto CLTest;
			}
#endif

			if(result.count("model")) args.modelType = result["model"].as<std::string>();
			string modelType = args.modelType;
			if (find_if(modelTypes.begin(), modelTypes.end(), [&modelType](const model_pair_t& it){ return it.first == modelType; }) == modelTypes.end())
				throw cxxopts::OptionException("Unknown model type: " + args.modelType);
			if (result.count("input")) args.input = result["input"].as<std::vector<std::string>>();

#define READ_OPT(P, TYPE) if (result.count(#P)) args.P = result[#P].as<TYPE>()
#define READ_OPT2(P, Q, TYPE) if (result.count(#P)) args.Q = result[#P].as<TYPE>()

			READ_OPT(load, string);
			READ_OPT(save, string);
			READ_OPT(inference, string);

			READ_OPT(verbose, int);

			READ_OPT(worker, int);
			READ_OPT2(topic, K, int);
			READ_OPT2(mdfield, numMDFields, int);
			READ_OPT2(maxline, numMaxLine, int);
			READ_OPT2(maxlen, numMaxLength, int);
			READ_OPT(seed, int);
			READ_OPT2(words, numTopWords, int);
			READ_OPT2(ws, numTopWordsSaving, int);
			READ_OPT(repeat, int);
			READ_OPT2(oi, optimInterval, int);
			READ_OPT2(or, optimRepeat, int);
			READ_OPT(bi, int);
			READ_OPT2(dei, dirichletEstIteration, int);
			READ_OPT(iteration, int);
			READ_OPT2(update, update, int);

			READ_OPT(alpha, double);
			READ_OPT(alphaEps, double);
			READ_OPT(eta, double);
			READ_OPT(gamma, double);
			READ_OPT(sigma, double);
			READ_OPT2(s0, sigma0, double);
			READ_OPT(lambda, double);

			READ_OPT(fa, int);
			READ_OPT(fI, int);
			READ_OPT2(mc, minCount, int);
			READ_OPT(rm, int);

			READ_OPT2(bsave, bestSave, string);
			READ_OPT2(tsave, saveTopicAssign, string);
			READ_OPT2(btsave, bestSaveTopicAssign, string);
			READ_OPT2(dsave, saveTopicDistByDoc, string);
			READ_OPT2(fullmodel, saveFullModel, int);
			READ_OPT2(wsave, saveWordDist, string);
			READ_OPT2(psave, saveParameters, string);
			READ_OPT(stopword, string);

			READ_OPT2(cl, clDevice, int);
			READ_OPT(clGroup, int);
			READ_OPT(clLocal, int);
			READ_OPT(clUpdate, int);
			
			if (!args.clUpdate) args.clUpdate = args.update;
			if (!args.alpha) args.alpha = 50. / args.K;
			if (!args.load.empty())
			{
				if (!result.count("iteration")) args.iteration = 0;
			}

			if (result.count("degree"))
			{
				args.degrees = stringToVector<size_t>(result["degree"].as<string>());
				if (!args.degrees.empty()) args.degree = args.degrees[0];
			}

			if (result.count("mdm"))
			{
				args.mdMin = stringToVector<float>(result["mdm"].as<string>());
			}
			if (result.count("mdM"))
			{
				args.mdMax = stringToVector<float>(result["mdM"].as<string>());
			}
		}
		catch (const cxxopts::OptionException& e) 
		{
			cout << "error parsing options: " << e.what() << endl;
			cout << options.help({ "" }) << endl;
			return -1;
		}

	}
	catch (const cxxopts::OptionException& e)
	{
		cout << "error parsing options: " << e.what() << endl;
		return -1;
	}
#ifdef OPENCL
CLTest:
	if ((args.clDevice || clList) && clLibLoad())
	{
		cout << "Failed to load OpenCL library. Retry without cl parameter!" << endl;
		return -1;
	}
	if (clList)
	{
		cout << "Available OpenCL devices in the environment" << endl;
		tomoto::ocl::OpenCLMgr clMgr;
		size_t num = 1;
		for (auto& device : clMgr.getDeviceList())
		{
			cout << num++ << ". " << device.name << '\t' << device.driver << '\t' << device.openCL << '\t' << device.version << '\t' << device.maxComputeUnits << endl;
		}
		return 0;
	}
#endif
	return newInstance(args.modelType)->run(args);
}
