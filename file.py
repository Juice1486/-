file=open('Hello.jpg','r')
file_write=open('Hello2.jpg','r')
while True:
    text=file.readline
    if not text:
        break
    file_write.write(text)
file.close()
file_write.close()
    
