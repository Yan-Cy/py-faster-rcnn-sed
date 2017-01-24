
def sortSet(setname):
    with open(setname) as f:
        setdata = [x.strip() for x in f.readlines()]
    setdata = sorted(setdata)
    with open(setname, 'w') as f:
        f.write('\n'.join(setdata))

def div2(setname):
    with open(setname) as f:
        setdata = [x.strip().split('_') for x in f.readlines()]
    setdata = sorted(setdata, key = lambda x: (x[0], x[1], x[2], x[3], int(x[4])))
    count = 0
    with open('div4' + setname, 'w') as f:
        for data in setdata:
            count += 1
            if count % 4 == 1:
                f.write('_'.join(data) + '\n')

if __name__ == '__main__':
    #sortSet('1206test.txt')
    div2('1113test.txt')

