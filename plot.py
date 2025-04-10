import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import argparse


sd15_all_blocks = {
    'conv_in': [
        'conv_in',
    ],
    'down_blocks[0]': [
        'down_blocks[0].resnets[0]',
        'down_blocks[0].attentions[0]',
        'down_blocks[0].resnets[1]',
        'down_blocks[0].attentions[1]',
        'down_blocks[0].downsamplers[0]',
    ],
    'down_blocks[1]': [
        'down_blocks[1].resnets[0]',
        'down_blocks[1].attentions[0]',
        'down_blocks[1].resnets[1]',
        'down_blocks[1].attentions[1]',
        'down_blocks[1].downsamplers[0]',
    ],
    'down_blocks[2]': [
        'down_blocks[2].resnets[0]',
        'down_blocks[2].attentions[0]',
        'down_blocks[2].resnets[1]',
        'down_blocks[2].attentions[1]',
        'down_blocks[2].downsamplers[0]',
    ],
    'down_blocks[3]': [
        'down_blocks[3].resnets[0]',
        'down_blocks[3].resnets[1]',
    ],
    'mid_block': [
        'mid_block.resnets[0]',
        'mid_block.attentions[0]',
        'mid_block.resnets[1]',
    ],
    'up_blocks[0]': [
        'up_blocks[0].resnets[0]',
        'up_blocks[0].resnets[1]',
        'up_blocks[0].upsamplers[0]',
    ],
    'up_blocks[1]': [
        'up_blocks[1].resnets[0]',
        'up_blocks[1].attentions[0]',
        'up_blocks[1].resnets[1]',
        'up_blocks[1].attentions[1]',
        'up_blocks[1].resnets[2]',
        'up_blocks[1].attentions[2]',
        'up_blocks[1].upsamplers[0]',
    ],
    'up_blocks[2]': [
        'up_blocks[2].resnets[0]',
        'up_blocks[2].attentions[0]',
        'up_blocks[2].resnets[1]',
        'up_blocks[2].attentions[1]',
        'up_blocks[2].resnets[2]',
        'up_blocks[2].attentions[2]',
        'up_blocks[2].upsamplers[0]',
    ],
    'up_blocks[3]': [
        'up_blocks[3].resnets[0]',
        'up_blocks[3].attentions[0]',
        'up_blocks[3].resnets[1]',
        'up_blocks[3].attentions[1]',
        'up_blocks[3].resnets[2]',
        'up_blocks[3].attentions[2]',
    ],
    'conv_out': [
        'conv_out',
    ]
}

blocks = [b for blocks_list in sd15_all_blocks.values() for b in blocks_list]



def plot(model_name):

    # ensure reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    plt.rcParams['svg.hashsalt'] = '42'

    # position estimation
    sizes_to_compare = [8, 16, 32, 64]
    accuracies_cls = torch.load(f'data/accuracies_cls_{model_name}.pt', weights_only=True)

    # group similarities (corners)
    results_group_similarities = torch.load(f'data/results_group_similarities_{model_name}.pt', weights_only=True)

    # up[1] anomalies
    up1_anomaly_norms_mean = torch.load(f'data/up1_anomaly_norms_mean_{model_name}.pt', weights_only=False)
    up1_anomaly_norms_std = torch.load(f'data/up1_anomaly_norms_std_{model_name}.pt', weights_only=False)


    ## plot
    # setup figure
    fig, axs = plt.subplots(3, 1, figsize=(8, 5))


    # compute main blocks names and positions
    main_blocks = []
    main_block_positions = []
    layer_counter = 0
    for block_name, block_list in sd15_all_blocks.items():
        if 'mid' in block_name:
            name = 'mid'
        elif 'conv' in block_name:
            name = block_name[5:]
        else:
            name = block_name.replace('_blocks','').replace('[','').replace(']','').replace('down', 'dn')
        main_blocks.append(name)
        main_block_positions.append(layer_counter)
        layer_counter += len(block_list)

    # lines between main blocks
    for p in main_block_positions[1:]:
        axs[0].axvline(x=p-0.5, color='black', linestyle='--', c='lightgray')
        axs[1].axvline(x=p-0.5, color='black', linestyle='--', c='lightgray')
        axs[2].axvline(x=p-0.5, color='black', linestyle='--', c='lightgray')
    ax_x3 = axs[2].secondary_xaxis(location=0)
    ax_x3.set_xticks([p-0.5 for p in main_block_positions[1:]], labels=[])
    ax_x3.tick_params(axis='x', length=34, width=1.5, color='lightgray')


    # plot accuracies
    blues = plt.cm.Blues(np.linspace(0.25, 0.9, len(sizes_to_compare)))
    reds = plt.cm.Reds(np.linspace(0.25, 0.9, len(sizes_to_compare)))
    for i in range(accuracies_cls.shape[2]):
        l1 = axs[0].plot(accuracies_cls[:, -1, i]*100, label=f'{sizes_to_compare[i]}×{sizes_to_compare[i]}', color=reds[i])

    # finish plot
    axs[0].set_xticks([])
    axs[0].set_xticklabels([])
    axs[0].legend(fontsize='small')
    axs[0].set_ylabel('test accuracy')
    axs[0].set_ylim(-5, 105)


    # plot results
    xs = range(len(blocks))
    for i, (group_name, color) in enumerate(zip(['corner', 'border'], ['tab:blue', '#50b0bf'])):
        axs[1].plot(xs[:-1], [x[0][i,i]/x[1][i,i] for x in results_group_similarities.values()][:-1], label=group_name, color=color)

    # reference line
    axs[1].plot(xs, [1]*len(xs), linestyle='--', color='gray')

    # configure plot
    axs[1].set_xticks([])
    axs[1].set_xticklabels([])
    axs[1].set_ylabel('rel. similarity')
    axs[1].legend(fontsize='small')
    axs[1].set_yscale('log')
    axs[1].set_yticks([0.5, 1, 2])
    axs[1].set_yticklabels([f'{x:.1f}' for x in axs[1].get_yticks()])
    axs[1].yaxis.set_minor_formatter(plt.NullFormatter())
    axs[1].set_ylim(0.46, 2.8)


    # up[1] anomaly
    axs[2].fill_between(range(len(blocks)),
                    up1_anomaly_norms_mean - up1_anomaly_norms_std,
                    up1_anomaly_norms_mean + up1_anomaly_norms_std, 
                    alpha=0.2, color='#26a259')
    axs[2].plot(range(len(blocks)), up1_anomaly_norms_mean, label='anomalies', color='#26a259')

    # mean norm (all, reference line)
    axs[2].plot(range(len(blocks)), np.ones(len(blocks)), color='gray', linestyle='--')

    # configure plot
    axs[2].set_ylabel("rel. norm")
    axs[2].legend(fontsize='small')
    axs[2].set_yscale('log')
    axs[2].set_yticks([0.5, 1.0, 2.0, 4.0])
    axs[2].set_yticklabels([f'{x:.1f}' for x in axs[2].get_yticks()])
    axs[2].yaxis.set_minor_formatter(plt.NullFormatter())
    axs[2].set_ylim(0.45, 5.8)


    # plot x ticks
    x = np.arange(len(blocks))
    ticks = ['attn' if 'attentions' in block else 'res' if 'resnets' in block else 'down' if 'downsamplers' in block else 'up' if 'upsamplers' in block else 'conv' if 'conv' in block else '?' for block in blocks]
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(ticks, rotation=90)
    axs[2].text(-0.01, -0.25, 'layers:', ha='right', va='center', transform=axs[2].transAxes)
    axs[2].text(-0.01, -0.45, 'blocks:', ha='right', va='center', transform=axs[2].transAxes)
    axs[0].set_xlim(-0.5, len(blocks))
    axs[1].set_xlim(-0.5, len(blocks))
    axs[2].set_xlim(-0.5, len(blocks))

    # plot main blocks names
    ax_x2 = axs[2].secondary_xaxis(location=0)
    ax_x2.set_xticks([p+len(bl)/2-0.5 for p, bl in zip(main_block_positions, sd15_all_blocks.values())], labels=[f'\n\n\n{b}' for b in main_blocks], ha='center')
    ax_x2.tick_params(length=0)

    # adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)

    # Make SVG text selectable and adjust settings
    # plt.rcParams['svg.fonttype'] = 'none'  # Make text selectable in SVG -> text size breaks in latex/pdf

    plt.savefig(f'results-quantitative-{model_name}.svg', bbox_inches='tight', format='svg', dpi=300, metadata={'Date': None})



def main():
    parser = argparse.ArgumentParser(description='Run SD representation experiments')
    parser.add_argument('--models', type=str, nargs='+', default=['sd15', 'sd21', 'sdturbo'], 
                        help='Models to analyze (default: sd15 sd21 sdturbo)')
    
    args = parser.parse_args()

    for model_name in args.models:
        print(f'Plotting {model_name}...')
        plot(model_name)



if __name__ == "__main__":
    main()
