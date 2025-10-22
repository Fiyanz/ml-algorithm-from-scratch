from ctypes.wintypes import tagPOINT

def desimal(x: int) -> float:
    return x / 100

def to_desimal(pa: int, pb: int, p_ax: int, p_bx: int) -> list:
    data = [pa, pb, p_ax, p_bx]
    return list(map(desimal, data))

def hitung_algo(data: list) -> float:
    pa, pb, p_ax, p_bx = data
    px = (pa * p_ax) + (pb * p_bx)
    result = p_ax * pa / px
    return result

def hitung_probabilitas(pa: int, pb: int, p_ax: int, p_bx: int) -> float:
    data_float = to_desimal(pa, pb, p_ax, p_bx)
    result = hitung_algo(data_float)
    return result

print(hitung_probabilitas(50, 50, 40, 10))
