
def sortSet(setname):
    with open(setname) as f:
        setdata = [x.strip() for x in f.readlines()]
    setdata = sorted(setdata)
    with open(setname, 'w') as f:
        f.write('\n'.join(setdata))

def div2(setname):
    with open(setname) as f:
        setdata = [(x.strip(), len(x.strip())) for x in f.readlines()]
    setdata = sorted(setdata, key = lambda x:x[1])
    count = 0
    with open('div2' + setname, 'w') as f:
        for data in setdata:
            count += 1
            if data[0][-2] != '_' and int(data[0][-2]) % 2 == 1:
                f.write(data[0] + '\n')

if __name__ == '__main__':
    #sortSet('1206test.txt')
    div2('1206test.txt')

