import tomotopy as tp
import numpy as np

def run_gdmr(input_file, topics, 
    degrees=[], optim_interval=20, burn_in=200, 
    alpha=1e-2, sigma=0.25, sigma0=3.0, md_range=None,
    iteration=800,
    save_path=None
    ):
    
    print(md_range)

    corpus = tp.utils.Corpus()
    for line in open(input_file, encoding='utf-8'):
        fd = line.strip().split()
        corpus.add_doc(fd[2:], metadata=list(map(float, fd[:2])))

    mdl = tp.GDMRModel(k=topics, degrees=degrees, alpha=alpha, sigma=sigma, sigma0=sigma0, 
        metadata_range=md_range, corpus=corpus
    )
    mdl.optim_interval = optim_interval
    mdl.burn_in = burn_in
    
    mdl.train(0)

    print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
    for i in range(0, iteration, 20):
        print('Iteration: {:04} LL per word: {:.4}'.format(i, mdl.ll_per_word))
        mdl.train(min(iteration - i, 20))
    print('Iteration: {:04} LL per word: {:.4}'.format(iteration, mdl.ll_per_word))

    if save_path: mdl.save(save_path)
    return mdl

def print_result(gdmr_mdl, visualize_map=False):
    if visualize_map:
        import matplotlib.pyplot as plt
        import matplotlib.colors as clr

        class ExpNormalize(clr.Normalize):
            def __init__(self, scale):
                super().__init__()
                self.scale = scale

            def __call__(self, value, clip=None):
                if clip is None:
                    clip = self.clip

                result, is_scalar = self.process_value(value)

                self.autoscale_None(result)
                (vmin,), _ = self.process_value(self.vmin)
                (vmax,), _ = self.process_value(self.vmax)
                if vmin == vmax:
                    result.fill(0)
                elif vmin > vmax:
                    raise ValueError("minvalue must be less than or equal to maxvalue")
                else:
                    if clip:
                        mask = np.ma.getmask(result)
                        result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                            mask=mask)
                    resdat = result.data
                    resdat = 1 - np.exp(-2 * resdat / self.scale)
                    result = np.ma.array(resdat, mask=result.mask, copy=False)
                if is_scalar:
                    result = result[0]
                return result
        
        heat = clr.LinearSegmentedColormap.from_list('heat', 
            [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0), (1, 1, 1)],
            N=1024
        )

    topic_counts = gdmr_mdl.get_count_by_topics()
    lambdas = gdmr_mdl.lambdas

    if visualize_map:
        md_range = gdmr_mdl.metadata_range
        r = gdmr_mdl.tdf_linspace(
            [md_range[0][0], md_range[1][0]], 
            [md_range[0][1], md_range[1][1]], 
            [400, 200]
        )

    for k in (-topic_counts).argsort():
        print('Topic #{} ({})'.format(k, topic_counts[k]))
        print(*(w for w, _ in gdmr_mdl.get_topic_words(k)))
        print('Lambda:', lambdas[k])

        if visualize_map:
            imgplot = plt.imshow(r[:, :, k].transpose(), clim=(0.0, r[:, :, k].max()), 
                origin='lower', cmap=heat, norm=ExpNormalize(scale=0.04),
                extent=[2000, 2017, 0, 1],
                aspect='auto'
            )
            plt.title('#{}\n({})'.format(k, ' '.join(w for w, _ in gdmr_mdl.get_topic_words(k, top_n=5))))
            plt.colorbar()
            plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str)
    parser.add_argument('--save', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('-K', '--topic', default=40, type=int)
    parser.add_argument('-F', '--degrees', nargs='*', type=int)
    parser.add_argument('--optim_interval', default=20, type=int)
    parser.add_argument('--burn_in', default=200, type=int)
    parser.add_argument('--iteration', default=800, type=int)
    parser.add_argument('--alpha', default=1e-2, type=float)
    parser.add_argument('--sigma', default=0.25, type=float)
    parser.add_argument('--sigma0', default=3.0, type=float)
    parser.add_argument('--md_range', default=None, type=str)
    parser.add_argument('--visualize', default=False, action='store_true')

    args = parser.parse_args()

    if args.load:
        mdl = tp.GDMRModel.load(args.load)
    elif args.input:
        mdl = run_gdmr(args.input, args.topic, args.degrees, 
            args.optim_interval, args.burn_in,
            args.alpha, args.sigma, args.sigma0,
            eval(args.md_range), args.iteration,
            args.save
        )
    else:
        raise ValueError("--load or --input should be given.")

    print_result(mdl, args.visualize)