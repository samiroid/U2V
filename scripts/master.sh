declare -a lrs=(10 1 0.1 )
declare -a margins=(10 5 1 )

for lr in "${lrs[@]}"
do
    for m in "${margins[@]}"
    do
        ./scripts/cmd_train_u2v.sh $lr $m     
        # exit 1   
    done
done