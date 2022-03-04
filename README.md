# GMM-MetropolisHastings-
Implementation of a Gaussian mixture model with Metropolis-Hastings algorithm.  

# How to run

1. The first step is to create the observation data using **make_data.py**. Then, create **data1.txt**. **true_label.txt** is the label data for calculating ARI.
2. After that, you can use **gmm_mh.py** to run the clustering.  

The image below shows the actual generated observables for the two modalities.（The cluster numbers for the two data points are the same）　　
<div>
	<img src='/image/data1.png' height="250px">
</div>

The image below shows the actual ARI measured by mgmm.py, where a value close to 1 means high cluster performance and a value close to 0 means low cluster performance.  

<div>
	<img src='/image/ari.png' height="250px">
</div>

<div>
	<img src='/image/accept.png' height="250px">
</div>