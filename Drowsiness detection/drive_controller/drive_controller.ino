void setup() {
  Serial.begin(9600);
  pinMode(6,OUTPUT);// put your setup code here, to run once:
  analogWrite(6,130);
}

void loop() {
  //if(Serial.available())
  //{
    char a=Serial.read();
    if(a=='1')
    {
      //for(int i=0;i<255;i+15)
      //{
        analogWrite(6,255);
      }
       if(a=='0')
       
    {
      analogWrite(6,130);
    }
  //}
  //else
   //analogWrite(6,130);


}
