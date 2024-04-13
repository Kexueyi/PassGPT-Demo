sign in as root
```sql
mysql -u root -p
```

create a new local user
```sql
CREATE USER 'username'@'localhost' IDENTIFIED BY 'your_password_here';
GRANT ALL PRIVILEGES ON testdb.* TO 'username'@'localhost';
FLUSH PRIVILEGES;
```
check users
```sql
SELECT User, Host FROM mysql.user;
```
