i=0
dev=0
for game in airraid boxing carnival pong mspacam spaceinvaders riverraid
do
  for folder in train test validation
  do
    if nvidia-smi | grep "Processes" > /dev/null ; then
      # No running processes found
      if (($dev < 8)); then
        echo "CUDA_VISIBLE_DEVICES=$dev python3 create_dataset.py -f $folder -g $game"
        CUDA_VISIBLE_DEVICES=$dev nohup python3 main.py --task eval --config configs/atari_$game.yaml --arch-type baseline seed 1 > nohup$i.out 2>&1 &
        dev=$((dev+1))
        i=$((i+1))
      else
        dev=0
        sleep 20
      fi
    else
      sleep 60
    fi
  done
done
