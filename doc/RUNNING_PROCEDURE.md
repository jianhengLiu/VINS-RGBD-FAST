# Dynamic-VINS testing procedure

1. ```
   export ROS_HOSTNAME=192.168.2.223     #从机IP 
   export ROS_MASTER_URI=http://192.168.2.223:11311
   roscore
   ```

2. ```
   export ROS_HOSTNAME=192.168.2.223     #从机IP 
   export ROS_MASTER_URI=http://192.168.2.223:11311
   rosrun image_transport republish raw in:=/d400/color/image_raw compressed out:=/d400/color/image_raw
   ```

3. ```
   export ROS_HOSTNAME=192.168.2.223     #从机IP 
   export ROS_MASTER_URI=http://192.168.2.223:11311
   rosrun image_transport republish raw in:=/d400/aligned_depth_to_color/image_raw compressedDepth out:=/d400/aligned_depth_to_color/image_raw
   ```

5. Dynamic-VINS

   1. run yolo, vins on atlas

      1. ```
         ssh HwHiAiUser@192.168.2.2
         export ROS_HOSTNAME=192.168.2.2     #从机IP
         export ROS_MASTER_URI=http://192.168.2.223:11311
         roslaunch vins_estimator nodelet_openloris.launch
         ```

   2. run yolo on atlas, and run vins on pc

       1. ```
            (run vins on pc)
            export ROS_HOSTNAME=192.168.2.223     #从机IP 
            export ROS_MASTER_URI=http://192.168.2.223:11311
            roslaunch vins_estimator nodelet_openloris_test.launch
            ```

6. rviz visualiztion

   ```
   export ROS_HOSTNAME=192.168.2.223     #从机IP 
   export ROS_MASTER_URI=http://192.168.2.223:11311
   roslaunch vins_estimator vins_rviz.launch 
   ```