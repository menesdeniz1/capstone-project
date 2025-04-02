#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <SoftwareSerial.h>

LiquidCrystal_I2C lcd(0x3f, 16, 2);
SoftwareSerial bluetooth(2, 3);

String gelenVeri = "";
String sonGosterilen = "";

void setup() {
  Serial.begin(9600);
  bluetooth.begin(9600);
  lcd.init();
  lcd.backlight();

  // Bluetooth bağlantısının kontrolü
  if (bluetooth.available()) {
    lcd.setCursor(0, 0);
    lcd.print("Bluetooth Hazir");
  } else {
    lcd.setCursor(0, 0);
    lcd.print("Bluetooth baglantisi yok");
  }
  delay(2000);
  lcd.clear();
}

void loop() {
  while (bluetooth.available()) {
    char karakter = bluetooth.read();

    // Satır sonu kontrolü
  if (karakter == '\n' || karakter == '\r') {
      if (gelenVeri.length() > 0) {
          Serial.print("Gelen Veri: ");
          Serial.println(gelenVeri);  // Gelen veriyi ekrana yazdırıyoruz.
          
          lcd.clear();
          lcd.setCursor(0, 0);
          lcd.print(gelenVeri);  // LCD'ye veriyi yazdırıyoruz.
          
          sonGosterilen = gelenVeri;
          gelenVeri = "";  // Yeni veriye hazırlık
      }
  } else {
      gelenVeri += karakter;
}


  }

  delay(100); // Döngüde biraz bekle
}
