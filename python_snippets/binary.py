import struct
import unicodedata as ud

blist = [1, 2, 3, 255]
the_bytes = bytes(blist)
print(the_bytes)
# the_bytes[1] = 127 тип bytes неизменяем
the_byte_array = bytearray(blist)
the_byte_array[1] = 255
# последовательность байтов должна быть в интервале [0,255]

# struct
valid_png_header = b'\x89PNG\r\n\x1a\n'
data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x9a\x00\x00\x00\x8d\x08\x02\x00\x00\x00\xc0'
if data[:8] == valid_png_header:
    width, height = struct.unpack('>LL', data[16:24]) # L - 4байтное число
    print('Valid PNG, width', width, 'height', height)
else:
    print('Not a valid PNG')

print( struct.pack('>L', 777) )

# hex to dec
print(int('ff', 16))
print(int(b'ff', 16))
print(int(b'0xff', 16))
print(int(b'\xff'.hex(), 16))


# информация по юникод символу
def unicode_info(un_symbol):
    ds = un_symbol.encode('utf-8')
    print(ds)
    print(ud.category(un_symbol),  ud.name(un_symbol), ud.lookup(ud.name(un_symbol)), sep='\n')
    print(un_symbol.encode('ascii','backslashreplace'))
    print(un_symbol.encode('ascii','xmlcharrefreplace'))

snowball = '\u2603'
unicode_info(snowball)