i=0
dev=0
for game in air_raid boxing carnival pong mspacam spaceinvaders riverraid
do
  for seed in 0 1 2 3 4
  do
    if nvidia-smi | grep "Processes" > /dev/null ; then
      # No running processes found
      if (($dev < 8)); then
        echo "CUDA_VISIBLE_DEVICES=$dev python3 main.py --task eval --config configs/atari_$game.yaml --arch-type baseline seed $seed"
        CUDA_VISIBLE_DEVICES=$dev nohup python3 main.py --task eval --config configs/atari_$game.yaml --arch-type baseline seed $seed > nohup$i.out 2>&1 &
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
