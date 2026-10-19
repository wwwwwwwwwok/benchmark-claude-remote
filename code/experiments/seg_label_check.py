"""Count frames that carry lidar 3D-segmentation labels and camera panoptic labels."""
import glob, os, struct, sys
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
FD = descriptor_pb2.FieldDescriptorProto
SCHEMA = {
    'RangeImage': [('segmentation_label_compressed', 6, FD.TYPE_BYTES, 0)],
    'Laser': [('name', 1, FD.TYPE_INT32, 0), ('ri_return1', 2, '.m.RangeImage', 0)],
    'CameraImage': [('name', 1, FD.TYPE_INT32, 0), ('camera_segmentation_label', 10, FD.TYPE_BYTES, 0)],
    'Frame': [('timestamp_micros', 2, FD.TYPE_INT64, 0), ('images', 4, '.m.CameraImage', 1), ('lasers', 5, '.m.Laser', 1)],
}
fdp = descriptor_pb2.FileDescriptorProto(name='m.proto', package='m', syntax='proto2')
for mn, fields in SCHEMA.items():
    m = fdp.message_type.add(name=mn)
    for fn, num, t, rep in fields:
        f = m.field.add(name=fn, number=num, label=FD.LABEL_REPEATED if rep else FD.LABEL_OPTIONAL)
        if isinstance(t, str):
            f.type, f.type_name = FD.TYPE_MESSAGE, t
        else:
            f.type = t
pool = descriptor_pool.DescriptorPool(); pool.Add(fdp)
Frame = message_factory.GetMessageClass(pool.FindMessageTypeByName('m.Frame'))

def records(path):
    with open(path, 'rb') as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            n = struct.unpack('<Q', head[:8])[0]
            data = f.read(n); f.read(4)
            yield data

for path in sorted(glob.glob(os.path.join(sys.argv[1], '*.tfrecord'))):
    n = lidar_seg = cam_seg = 0
    cams = set()
    for rec in records(path):
        fr = Frame(); fr.ParseFromString(rec); n += 1
        if any(l.name == 1 and len(l.ri_return1.segmentation_label_compressed) > 0 for l in fr.lasers):
            lidar_seg += 1
        hit = [im.name for im in fr.images if len(im.camera_segmentation_label) > 0]
        if hit:
            cam_seg += 1; cams.update(hit)
    print(os.path.basename(path)[8:45], f'frames {n}, with lidar seg labels {lidar_seg}, with camera panoptic labels {cam_seg} (cameras {sorted(cams)})', flush=True)
