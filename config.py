class Models:
    def __init__(self):
        self.protoFile_coco = "pose/coco/pose_deploy_linevec.prototxt"
        self.weightsFile_coco = "pose/coco/pose_iter_440000.caffemodel"
        self.nPoints_coco = 18
        self.pose_pairs_coco = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [
            8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

        self.protoFile_mpi = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        self.weightsFile_mpi = "pose/mpi/pose_iter_160000.caffemodel"
        self.nPoints_mpi = 15
        self.pose_pairs_mpi = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [
            6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]
