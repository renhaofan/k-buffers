for SCENE in plant resume washing;
do
    cd /home/renhaofan/dataset/high-freq-dataset/$SCENE
    cp -r images images_4
    cd images_4
    mogrify -resize 25% *.jpg

    cd /home/renhaofan/dataset/high-freq-dataset/$SCENE
    cp -r images images_2
    cd images_2
    mogrify -resize 50% *.jpg

    cd /home/renhaofan/dataset/high-freq-dataset/$SCENE
done
