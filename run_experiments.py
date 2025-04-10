import numpy as np
import matplotlib.pyplot as plt
import torch
from sdhelper import SD
from datasets import load_from_disk
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse


# Define SD15 blocks structure
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


def run_position_estimation(model_name, blocks, representations, num_epochs=5, sizes_to_compare=[8, 16, 32, 64]):
    """Run position estimation experiment"""
    print("Running position estimation experiment...")
    
    # train config
    batch_size = 512
    num_train = int(len(representations) * 0.8)
    
    # set seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # setup logging variables
    classifiers = []
    accuracies_cls = torch.full((len(blocks), num_epochs, len(sizes_to_compare)), torch.nan)
    
    # train models
    for block_idx, block in enumerate(tqdm(blocks)):
        # setup models
        _, features, w, h = representations[0][block].shape
        classifier = nn.Linear(features, w+h).cuda()
        opt2 = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        classifiers.append(classifier)
    
        # get representations
        reprs = torch.stack([r[block].squeeze(0) for r in representations]).permute(0, 2, 3, 1).flatten(0, 2).cuda()
        labels = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(h), indexing='ij'), dim=-1).expand(len(representations), -1, -1, -1).flatten(0, 2).cuda()
        reprs_train = reprs[:num_train*w*h]
        labels_train = labels[:num_train*w*h]
        reprs_test = reprs[num_train*w*h:]
        labels_test = labels[num_train*w*h:]
    
        # epoch loop
        for epoch in range(num_epochs):
            # train
            classifier.train()
            indices = torch.randperm(len(reprs_train))
            for i in range(0, len(reprs_train), batch_size):
                reprs_batch = reprs_train[indices[i:i+batch_size]].float()
                labels_batch = labels_train[indices[i:i+batch_size]]
    
                classifier.zero_grad()
                preds = classifier(reprs_batch).view(-1, w, 2)
                loss = nn.functional.cross_entropy(preds, labels_batch)
                loss.backward()
                opt2.step()
    
            # test at epoch end
            with torch.no_grad():
                classifier.eval()
                preds_cls = classifier(reprs_test.float()).view(len(reprs_test), w, 2)
                loss_cls = nn.functional.cross_entropy(preds_cls, labels_test)
                for i, comparison_size in enumerate(sizes_to_compare):
                    accuracies_cls[block_idx, epoch, i] = (preds_cls.argmax(dim=1)*comparison_size // w == labels_test*comparison_size // w).float().mean().cpu()
    
    # Save results
    os.makedirs('data', exist_ok=True)
    torch.save(accuracies_cls, f'data/accuracies_cls_{model_name}.pt')
    print(f"Position estimation results saved to data/accuracies_cls_{model_name}.pt")
    return accuracies_cls


def run_token_group_similarities(model_name, blocks, images, sd, limit_images=500):
    """Run token group similarities experiment"""
    print("Running token group similarities experiment...")
    
    # setup slices to select groups of tokens
    corner_slices = [(a,b) for a in [0,-1] for b in [0,-1]]
    border_slices = [(slice(1,-1), 0), (slice(1,-1), -1), (0, slice(1,-1)), (-1, slice(1,-1))]
    other_slices  = [(slice(1,-1), slice(1,-1))]
    slices = [
        ('corner', corner_slices),
        ('border', border_slices),
        ('other', other_slices),
    ]
    
    # helper function to calculate similarities between the tokens in a group
    def calc_group_similarities(block, images):
        w, h = images[0].size
        representations = sd.img2repr(images[:limit_images], extract_positions=[block], step=50, seed=42)
        token_size = w // representations[0][block].shape[-1]
        representations_cropped = sd.img2repr([img.crop((token_size, token_size, w-token_size, h-token_size)) for img in images[:limit_images]], extract_positions=[block], step=50, seed=42)
        similarity_maps = torch.stack([x.cosine_similarity(x) for x in representations])
        similarity_maps_cropped = torch.stack([x.cosine_similarity(x) for x in representations_cropped])
        similarity_maps_repr_cropped = F.pad(similarity_maps, [-1]*8, value=torch.nan)
    
        results = []
        for sim_maps in [similarity_maps_cropped, similarity_maps_repr_cropped]:
            n = similarity_maps.shape[1]
            m = len(slices)
            result = torch.zeros((m, m))
            for i, (_, slices1) in enumerate(slices):
                for j, (_, slices2) in enumerate(slices):
                    count = torch.stack([torch.ones((n,n,n,n))[s11,s12,s21,s22].sum() for s11, s12 in slices1 for s21, s22 in slices2]).sum() * len(sim_maps)
                    self_similarities = torch.stack([torch.ones((n,n))[s11,s12].sum() for s11, s12 in slices1]).sum() * len(sim_maps) if i == j else 0
                    result[i,j] = (torch.stack([sim_maps[:,s11,s12,s21,s22].sum() for s11, s12 in slices1 for s21, s22 in slices2]).sum() - self_similarities) / (count - self_similarities)
            results.append(result)
        return results
    
    # calculate similarities
    results_group_similarities = {}
    for block in tqdm(blocks):
        results_group_similarities[block] = calc_group_similarities(block, images)
    
    # Save results
    os.makedirs('data', exist_ok=True)
    torch.save(results_group_similarities, f'data/results_group_similarities_{model_name}.pt')
    print(f"Token group similarities results saved to data/results_group_similarities_{model_name}.pt")
    return results_group_similarities


def run_anomaly_analysis(model_name, blocks, images, sd):
    """Run anomaly analysis experiment"""
    print("Running anomaly analysis experiment...")
    
    # load anomalies data
    up1_anomalies = np.load(f"labeler/high_norm_anomalies_{model_name}.npy")
    
    # get representations
    representations_raw = sd.img2repr(images, extract_positions=blocks, step=50, seed=42)
    
    # init result matrices
    up1_anomaly_norms = np.zeros((len(blocks), len(up1_anomalies)))
    
    # up1 anomaly
    for j, tmp in enumerate(tqdm(up1_anomalies)):
        img_idx, w_idx, h_idx = tmp.tolist()
        for i, block in enumerate(blocks):
            repr = representations_raw[img_idx][block].squeeze(0).to(dtype=torch.float32)
            features, h, w = repr.shape
            norms = repr.norm(dim=0)
            h_up1 = representations_raw[0]['up_blocks[1].upsamplers[0]'].shape[2]
            scale = h / h_up1
            h_idx_scaled = int(h_idx*scale)
            w_idx_scaled = int(w_idx*scale)
            offset = int(2*scale)
            if offset < 1: offset = 1
            reprs_anomaly = repr[:,h_idx_scaled:h_idx_scaled+offset, w_idx_scaled:w_idx_scaled+offset]
            rel_norm = reprs_anomaly.norm(dim=0).mean() / norms.mean()
            up1_anomaly_norms[i,j] = rel_norm.item()
    
    # save norm maps
    representation_norms = {b: torch.stack([x[b][0].to(dtype=torch.float32).norm(dim=0) for x in representations_raw]) for b in tqdm(blocks)}
    
    # Save results
    os.makedirs('data', exist_ok=True)
    torch.save(representation_norms, f'data/representation_norms_{model_name}.pt')
    torch.save(up1_anomaly_norms.mean(axis=1), f'data/up1_anomaly_norms_mean_{model_name}.pt')
    torch.save(up1_anomaly_norms.std(axis=1), f'data/up1_anomaly_norms_std_{model_name}.pt')
    print(f"Anomaly analysis results saved to data/")
    return up1_anomaly_norms, representation_norms


def main():
    parser = argparse.ArgumentParser(description='Run SD representation experiments')
    parser.add_argument('--models', type=str, nargs='+', default=['sd15', 'sd21', 'sdturbo'], 
                        help='Models to analyze (default: sd15 sd21 sdturbo)')
    parser.add_argument('--experiments', type=str, nargs='+', default=['position', 'token_groups', 'anomalies'], 
                        help='Experiments to run (default: all)')
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Please use a GPU for these experiments.")
        return
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Load data
    data = load_from_disk("imagenet_subset")
    images = [d['image'] for d in data]
    blocks = [b for blocks_list in sd15_all_blocks.values() for b in blocks_list]
    
    # Run experiments for each model
    for model_name in args.models:
        print(f"\nRunning experiments for model: {model_name}")
        
        # Load model
        sd = SD(model_name, disable_progress_bar=True)
        
        # Get representations for all experiments
        print("Extracting representations...")
        representations = sd.img2repr(images, blocks, 50, seed=42)
        
        # Run selected experiments
        if 'position' in args.experiments:
            run_position_estimation(model_name, blocks, representations)
        
        if 'token_groups' in args.experiments:
            run_token_group_similarities(model_name, blocks, images, sd)
        
        if 'anomalies' in args.experiments:
            run_anomaly_analysis(model_name, blocks, images, sd)
    
    print("\nAll experiments completed!")



if __name__ == "__main__":
    main()
