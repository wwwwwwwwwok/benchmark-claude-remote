"""Read-only (user, 2026-11-06, E item 8): do the Waymo Open Dataset protos give traffic signal states in a Perception
frame? The official package waymo-open-dataset-tf-2-12-0 1.6.5 (in the gssm_perception env; newer than the v1.4.3 data,
so any field of v1.4.3 is in it): every field of dataset_pb2.Frame and of the messages it contains (recursively, with
their types), and where map_pb2's traffic-signal messages are used. Imports the generated *_pb2 modules only.
usage: python -I proto_fields.py <site-packages folder>"""
import importlib
import os
import sys

SP = sys.argv[1]
sys.path.insert(0, SP)
pkg = os.path.join(SP, 'waymo_open_dataset')
mods = {}
for sub, names in (('', ['dataset_pb2', 'label_pb2']), ('protos', ['map_pb2', 'scenario_pb2', 'vector_pb2'])):
    for n in names:
        full = 'waymo_open_dataset.' + (sub + '.' if sub else '') + n
        try:
            mods[n] = importlib.import_module(full)
        except Exception as e:  # noqa: BLE001
            print(f'{full}: not imported ({type(e).__name__}: {str(e)[:150]})')
seen = set()


def walk(desc, depth=0, path='Frame'):
    if desc.full_name in seen:
        return
    seen.add(desc.full_name)
    for f in desc.fields:
        t = f.message_type.full_name if f.message_type else f.type
        rep = ' (repeated)' if f.label == f.LABEL_REPEATED else ''
        print('  ' * depth + f'{path}.{f.name} = {f.number}: {t}{rep}')
        if f.message_type and depth < 2:
            walk(f.message_type, depth + 1, f'{path}.{f.name}')


print('== dataset_pb2.Frame (fields of contained messages to depth 2)')
walk(mods['dataset_pb2'].Frame.DESCRIPTOR)
print('\n== messages in map_pb2 / scenario_pb2 whose name mentions a signal or a dynamic state, and their users')
for n in ('map_pb2', 'scenario_pb2'):
    fd = mods[n].DESCRIPTOR
    for mname, md in fd.message_types_by_name.items():
        if any(k in mname.lower() for k in ('signal', 'dynamic')):
            print(f'  {n}.{mname}: fields {[f.name for f in md.fields]}')
users = []
for n, m in mods.items():
    for mname, md in m.DESCRIPTOR.message_types_by_name.items():
        for f in md.fields:
            if f.message_type and any(k in f.message_type.name.lower() for k in ('signal', 'dynamicstate', 'dynamicmap')):
                users.append(f'{n}.{mname}.{f.name} -> {f.message_type.full_name}')
print('  fields of those types: ' + ('; '.join(users) if users else 'none'))
frame_names = {d for d in seen}
print(f'\nany message reachable from Frame with signal / dynamic in its name: '
      f'{sorted(x for x in frame_names if "signal" in x.lower() or "dynamic" in x.lower()) or "none"}')
