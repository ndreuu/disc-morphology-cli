"""Unreviewed posterior-body reference, independent of candidate lesion masks."""
import math

def _polygons(mask):
    """Trace all native pixel-cell boundary loops; no contour smoothing."""
    edges = {}
    rows, cols = mask.shape
    for y, x in zip(*mask.nonzero()):
        x, y = int(x), int(y)
        # Twice-native coordinates represent half-pixel edges exactly.
        corners = [(2*x-1,2*y-1),(2*x+1,2*y-1),(2*x+1,2*y+1),(2*x-1,2*y+1)]
        for i, (dy, dx) in enumerate(((-1,0),(0,1),(1,0),(0,-1))):
            yy, xx = y+dy, x+dx
            if not (0 <= yy < rows and 0 <= xx < cols and mask[yy,xx]):
                edges.setdefault(corners[i], []).append(corners[(i+1)%4])
    loops = []
    while edges:
        start = next(iter(edges)); point = start; loop = []
        while True:
            loop.append([point[0]/2,point[1]/2])
            options = edges.get(point)
            if not options:
                break
            target = options.pop()
            if not options:
                del edges[point]
            point = target
            if point == start:
                break
        if len(loop) >= 3:
            loops.append(loop)
    return sorted(loops, key=len, reverse=True)


def _world(frame, xy):
    import numpy as np
    from ._spatial import pixel_to_lps
    return np.asarray(pixel_to_lps(frame, *xy))*np.asarray([-1,-1,1])


def _reference(sagittal, anatomy, affine, label):
    """Fit posterior Y = a*X+b*Z+c from adjacent-body endplate strips.

    Connected horizontal runs must contain the disc's AP centre, avoiding the
    posterior elements of the complete vertebral labels. Every eligible native
    sagittal frame contributes; the lesion mask never selects reference points.
    """
    import numpy as np
    from .mask_io import project_to_frame
    upper, lower = {92:(41,42),93:(42,43),94:(43,44),95:(44,45),100:(45,50)}[label]
    disc_voxels=np.argwhere(anatomy==label)
    if len(disc_voxels)<12:
        raise ValueError('Anatomy has no resolved target disc')
    disc_center=affine@np.append(np.median(disc_voxels,axis=0),1)
    points, trace = [], []
    for frame in sagittal:
        labels = project_to_frame(anatomy, affine, frame)
        disc = np.argwhere(labels == label)
        if len(disc) < 12:
            continue
        cx = int(np.median(disc[:,1]))
        # Use central body contours rather than lateral pedicles. The fixed
        # physical window depends on anatomy alone, never on the lesion mask.
        center_xy=[float(cx),float(np.median(disc[:,0]))]
        if abs(_world(frame,center_xy)[0]-disc_center[0])>10:
            continue
        if (disc[:,1].max()-disc[:,1].min())*frame['spacing'][1] < 15:
            continue
        window = max(1, int(round(5/frame['spacing'][1])))
        xx = np.arange(frame['columns'])[None,:]
        for vertebra, direction in ((upper,-1),(lower,1)):
            body = np.argwhere((labels == vertebra) & (abs(xx-cx)<=window))
            if len(body) < 12:
                continue
            ey = int(body[:,0].max() if direction == -1 else body[:,0].min())
            endpoints = []
            for offset in (1,2,3):
                y = ey+direction*offset
                if not 0<=y<frame['rows'] or labels[y,cx] != vertebra:
                    continue
                end = cx
                while end+1<frame['columns'] and labels[y,end+1] == vertebra:
                    end += 1
                endpoints.append(end)
            if endpoints:
                xy = [float(np.median(endpoints)),float(ey)]
                world = _world(frame,xy)
                points.append(world)
                trace.append({'image_ref':frame['image_ref'],'vertebral_label':vertebra,
                              'point_xy':xy,'point_ras_mm':world.tolist()})
    if len(points) < 6:
        raise ValueError('Insufficient automatic posterior-body reference points')
    points = np.asarray(points)
    design = np.column_stack((points[:,0],points[:,2],np.ones(len(points))))
    coeff, _, rank, _ = np.linalg.lstsq(design,points[:,1],rcond=None)
    if rank != 3:
        raise ValueError('Posterior reference points cannot determine a plane')
    residual = float(np.sqrt(np.mean((design@coeff-points[:,1])**2)))
    return {'method':'posterior_body_endplate_plane_heuristic',
            'coefficients_y_from_x_z':coeff.tolist(), 'fit_rms_mm':residual,
            'points':trace, 'reviewed':False,
            'central_sagittal_half_window_mm':10,
            'limitations':['flat_plane_not_ring_apophysis_contour',
                           'osteophytes_not_independently_segmented']}


def _line(frame, reference):
    import numpy as np
    a,b,c = reference['coefficients_y_from_x_z']
    def value(x,y):
        world = _world(frame,[x,y])
        return float(a*world[0]+b*world[2]+c-world[1])
    v0 = value(0,0); vx = value(1,0)-v0; vy = value(0,1)-v0
    crossings = []
    w,h = frame['columns']-1,frame['rows']-1
    for p,q in (((0,0),(w,0)),((w,0),(w,h)),((w,h),(0,h)),((0,h),(0,0))):
        fp,fq = value(*p),value(*q)
        if fp == 0:
            crossings.append(list(p))
        if fp*fq < 0:
            t=fp/(fp-fq)
            crossings.append([p[i]+t*(q[i]-p[i]) for i in range(2)])
    if len(crossings)<2 or math.hypot(vx,vy)<1e-9:
        raise ValueError('Reference plane does not intersect the native frame')
    center = np.mean(np.asarray(crossings[:2]),axis=0)
    # Increasing signed plane value points posteriorly, toward negative RAS Y.
    normal=np.asarray([vx,vy]); normal/=np.linalg.norm(normal)
    return crossings[:2], (center+normal*10).tolist()
