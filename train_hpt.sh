loss_funcs=("nll_loss" "RMSE" "target_only" "dot_loss")
bs=(1 4 8 16 32)
lr=(0.001 0.01 0.1)
for loss in ${loss_funcs[@]}
do 
	for b in ${bs[@]}
	do
		for l in ${lr[@]}
		do
			echo $loss
		done
	done
done
