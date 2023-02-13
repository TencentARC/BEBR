import numpy as np
import argparse

def compute_similarity(image_features, text_features, bs = 1000):
    # compute similarity
    max_pairs = image_features.shape[0]
    similarity_scores = np.zeros((max_pairs, max_pairs), dtype=np.float32)
    for v in range(0, max_pairs, bs):
        for t in range(0, max_pairs, bs):
            print('Processing Visual '+str(v)+' Text '+str(t), end='\r')
            batch_visual_emb = image_features[v:v+bs]
            batch_caption_emb = text_features[t:t+bs]

            logits = batch_visual_emb @ np.transpose(batch_caption_emb)
            similarity_scores[v:v+bs,t:t+bs] = logits

    print('Done similarity')
    return similarity_scores

def compute_retrieval(a2b_sims, return_ranks=True):
    """
    Args:
        a2b_sims: Result of computing similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # loop source embedding indices
    for index in range(npts):
        # get order of similarities to target embeddings
        inds = np.argsort(a2b_sims[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    report_dict = {"r1": r1, "r5": r5, "r10": r10, "r50": r50, "medr": medr, "meanr": meanr, "sum": r1 + r5 + r10}

    if return_ranks:
        return report_dict, (ranks, top1)
    else:
        return report_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_feat', type=str, required=True, help='path to image feature file')
    parser.add_argument('--txt_feat', type=str, required=True, help='path to text feature file')

    args, _ = parser.parse_known_args()

    image_feat = np.load(args.image_feat)
    txt_feat = np.load(args.txt_feat)

    image_feat = image_feat / np.linalg.norm(image_feat,axis=1)[:,np.newaxis]
    txt_feat = txt_feat / np.linalg.norm(txt_feat,axis=1)[:,np.newaxis]

    similarity_scores = compute_similarity(image_feat, txt_feat)
    i2t_dict = compute_retrieval(similarity_scores)
    print(i2t_dict)
